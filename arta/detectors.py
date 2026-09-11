"""Enhanced ECG-only GRIP-IBI algorithm.

This file intentionally lives next to, but separate from, ``core_graph.py``.
Use the previous GRIP with:

    .venv/bin/python HRV/step2_grip_ibi/core_graph.py

Use this enhanced GRIP variant with:

    .venv/bin/python HRV/step2_grip_ibi/detectors.py

Algorithm idea
--------------
The first GRIP implementation competed at the candidate-detector level using
only Pan-Tompkins + wavelet candidates. This enhanced version uses a stronger
candidate pool and lets GRIP operate where it is more useful: robust path
repair.

1. Use WFDB-XQRS as a high-confidence ECG-only backbone.
2. Generate extra candidates with Pan-Tompkins and wavelet detectors.
3. Prune likely false positives from the XQRS backbone when they create
   suspiciously short RR intervals relative to the local rhythm.
4. Fill only obvious long gaps using the extra candidate graph.

No PPG, accelerometer, or BPM0 labels are used by the detector.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib_cache"))

import numpy as np
from wfdb import processing as wfdb_processing

try:
    from .core_graph import (
        FS,
        RR_MIN_MS,
        RR_MAX_MS,
        bandpass_filter,
        compare_to_bpm0,
        compute_bpm_windows,
        estimate_prior_hr,
        flag_ibi_outliers,
        merge_candidates_nms,
        pan_tompkins_candidates,
        plot_bland_altman,
        plot_bpm_timeline,
        plot_example_peaks,
        plot_scatter,
        save_peak_result,
        summarize_beat_rows,
        summarize_window_rows,
        wavelet_candidates,
        write_csv,
        json_safe,
        load_npz_scalar,
        match_rpeaks,
        ibi_error_from_matched_peaks,
        group_rows,
    )
except ImportError:  # Allows direct execution from this directory.
    from core_graph import (
        FS,
        RR_MIN_MS,
        RR_MAX_MS,
        bandpass_filter,
        compare_to_bpm0,
        compute_bpm_windows,
        estimate_prior_hr,
        flag_ibi_outliers,
        merge_candidates_nms,
        pan_tompkins_candidates,
        plot_bland_altman,
        plot_bpm_timeline,
        plot_example_peaks,
        plot_scatter,
        save_peak_result,
        summarize_beat_rows,
        summarize_window_rows,
        wavelet_candidates,
        write_csv,
        json_safe,
        load_npz_scalar,
        match_rpeaks,
        ibi_error_from_matched_peaks,
        group_rows,
    )

import json
import matplotlib

matplotlib.use("Agg")


LOGGER = logging.getLogger("detector_repair_ibi")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = np.percentile(values, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return np.ones(values.size, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def wfdb_xqrs_candidates(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """Run WFDB-XQRS as a high-recall/high-confidence ECG-only backbone."""
    return np.asarray(
        wfdb_processing.xqrs_detect(sig=np.asarray(ecg, dtype=np.float64), fs=fs, verbose=False),
        dtype=np.int64,
    )


def local_peak_quality(ecg: np.ndarray, peaks: np.ndarray, fs: float = FS) -> np.ndarray:
    """QRS-like local energy quality used to decide which close peak to prune."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        return np.array([], dtype=np.float32)
    qrs_band = np.abs(bandpass_filter(ecg, fs, 0.5, 40.0, order=3))
    safe_peaks = np.clip(peaks, 0, qrs_band.size - 1)
    return normalize(qrs_band[safe_peaks])


def extra_candidate_pool(ecg: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Generate non-XQRS candidates for filling likely missed beats."""
    pt_peaks, pt_scores = pan_tompkins_candidates(ecg, fs)
    wav_peaks, wav_scores = wavelet_candidates(ecg, fs)
    return merge_candidates_nms(
        [
            ("pan_tompkins", pt_peaks, pt_scores),
            ("wavelet", wav_peaks, wav_scores),
        ],
        ecg,
        fs,
        nms_ms=90.0,
    )


def prune_short_interval_false_peaks(
    peaks: np.ndarray,
    ecg: np.ndarray,
    fs: float = FS,
    short_fraction: float = 0.60,
    passes: int = 5,
) -> np.ndarray:
    """Remove likely false positives that create too-short local RR intervals.

    The rule is deliberately conservative: compare each RR interval with the
    local 7-interval median, and remove only the lower-quality peak in a close
    pair. This targets motion-artifact spikes that mimic extra beats.
    """
    peaks = np.asarray(sorted(set(map(int, peaks))), dtype=np.int64)
    if peaks.size < 5:
        return peaks

    for _ in range(max(1, passes)):
        rr_ms = np.diff(peaks) / float(fs) * 1000.0
        qualities = local_peak_quality(ecg, peaks, fs)
        remove: list[int] = []

        for idx, rr in enumerate(rr_ms):
            start = max(0, idx - 3)
            stop = min(rr_ms.size, idx + 4)
            local_median = float(np.median(rr_ms[start:stop]))
            threshold = max(RR_MIN_MS, short_fraction * local_median)
            if rr < threshold:
                # Drop the lower-quality member of the too-close pair.
                # Ties are deterministic: remove the later peak.
                remove.append(idx if qualities[idx] < qualities[idx + 1] else idx + 1)

        if not remove:
            break

        keep = np.ones(peaks.size, dtype=bool)
        for idx in sorted(set(remove)):
            if 0 <= idx < keep.size:
                keep[idx] = False
        new_peaks = peaks[keep]
        if new_peaks.size == peaks.size:
            break
        peaks = new_peaks

    return peaks


def fill_long_gap_missed_beats(
    backbone_peaks: np.ndarray,
    fill_candidates: np.ndarray,
    fill_scores: np.ndarray,
    fs: float = FS,
    long_fraction: float = 1.80,
    search_radius_s: float = 0.18,
) -> np.ndarray:
    """Insert candidate beats in obvious long gaps in the backbone path."""
    peaks = np.asarray(sorted(set(map(int, backbone_peaks))), dtype=np.int64)
    fill_candidates = np.asarray(fill_candidates, dtype=np.int64)
    fill_scores = np.asarray(fill_scores, dtype=np.float64)
    if peaks.size < 5 or fill_candidates.size == 0:
        return peaks

    rr_ms = np.diff(peaks) / float(fs) * 1000.0
    inserts: list[int] = []
    search_radius = int(round(search_radius_s * fs))

    for idx, rr in enumerate(rr_ms):
        local = rr_ms[max(0, idx - 3) : min(rr_ms.size, idx + 4)]
        expected_rr = float(np.median(local))
        if expected_rr <= 0:
            continue
        if rr <= min(RR_MAX_MS, long_fraction * expected_rr):
            continue

        n_missing = int(round(rr / expected_rr) - 1)
        if n_missing <= 0 or n_missing > 3:
            continue

        left_peak = peaks[idx]
        right_peak = peaks[idx + 1]
        for missing_idx in range(1, n_missing + 1):
            target = left_peak + int(round((right_peak - left_peak) * (missing_idx / (n_missing + 1))))
            lo = target - search_radius
            hi = target + search_radius
            mask = (fill_candidates >= lo) & (fill_candidates <= hi)
            if np.any(mask):
                local_candidates = fill_candidates[mask]
                local_scores = fill_scores[mask]
                inserts.append(int(local_candidates[int(np.argmax(local_scores))]))

    if inserts:
        peaks = np.asarray(sorted(set(peaks.tolist() + inserts)), dtype=np.int64)
    return peaks


def estimate_rpeaks_detector_repair(
    ecg: np.ndarray,
    fs: float = FS,
    short_fraction: float = 0.60,
    long_fraction: float = 1.80,
) -> dict[str, object]:
    """Estimate R-peaks with XQRS-seeded graph/rhythm repair."""
    ecg = np.asarray(ecg, dtype=np.float64).squeeze()

    xqrs_peaks = wfdb_xqrs_candidates(ecg, fs)
    fill_candidates, fill_scores = extra_candidate_pool(ecg, fs)

    pruned = prune_short_interval_false_peaks(
        xqrs_peaks,
        ecg,
        fs=fs,
        short_fraction=short_fraction,
    )
    repaired = fill_long_gap_missed_beats(
        pruned,
        fill_candidates,
        fill_scores,
        fs=fs,
        long_fraction=long_fraction,
    )

    rpeaks = np.asarray(sorted(set(map(int, repaired))), dtype=np.int64)
    ibi_ms = np.diff(rpeaks) / float(fs) * 1000.0
    instant_bpm = 60000.0 / ibi_ms if ibi_ms.size else np.array([], dtype=np.float64)
    outlier_mask = flag_ibi_outliers(ibi_ms)
    prior_centers_s, prior_hr_bpm, prior_ibi_ms = estimate_prior_hr(rpeaks, ecg.size, fs)

    return {
        "rpeaks_samples": rpeaks.astype(np.int64),
        "rpeaks_seconds": (rpeaks / float(fs)).astype(np.float32),
        "ibi_ms": ibi_ms.astype(np.float32),
        "instant_bpm": instant_bpm.astype(np.float32),
        "ibi_outlier_mask": outlier_mask,
        "candidate_peaks": fill_candidates.astype(np.int64),
        "candidate_scores": fill_scores.astype(np.float32),
        "prior_centers_s": prior_centers_s,
        "prior_hr_bpm": prior_hr_bpm,
        "prior_ibi_ms": prior_ibi_ms,
        "n_xqrs_backbone_candidates": int(xqrs_peaks.size),
        "n_fill_candidates": int(fill_candidates.size),
        "n_pruned_from_xqrs": int(max(0, xqrs_peaks.size - pruned.size)),
        "n_inserted_from_fill_candidates": int(max(0, rpeaks.size - pruned.size)),
        # Compatibility with save_peak_result from the original GRIP runner.
        "n_pan_tompkins_candidates": int(fill_candidates.size),
        "n_wavelet_candidates": int(fill_candidates.size),
        "n_merged_candidates": int(fill_candidates.size),
        "beam_width": 0,
        "sigma_hr_ms": math.nan,
        "short_fraction": float(short_fraction),
        "long_fraction": float(long_fraction),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    """Run enhanced GRIP alone and save the same core reports as old GRIP."""
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_files = sorted((processed_dir / "noisy_records").glob("snr_*db/*.npz"))
    if args.snrs is not None:
        requested = {float(snr) for snr in args.snrs}
        kept = []
        for path in noisy_files:
            with np.load(path, allow_pickle=True) as data:
                if float(load_npz_scalar(data, "target_snr_db", math.nan)) in requested:
                    kept.append(path)
        noisy_files = kept
    if args.max_files is not None:
        noisy_files = noisy_files[: args.max_files]
    if not noisy_files:
        raise FileNotFoundError(f"No noisy records found under {processed_dir / 'noisy_records'}")

    reference_cache: dict[str, dict[str, object]] = {}
    record_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    beat_rows: list[dict[str, object]] = []

    for path in noisy_files:
        with np.load(path, allow_pickle=True) as data:
            noisy_ecg = np.asarray(data["noisy_ecg"], dtype=np.float32)
            clean_ecg = np.asarray(data["clean_ecg"], dtype=np.float32)
            bpm0 = np.asarray(data["bpm0"], dtype=np.float32)
            window_start_s = np.asarray(data["window_start_s"], dtype=np.float32)
            window_end_s = np.asarray(data["window_end_s"], dtype=np.float32)
            fs = float(load_npz_scalar(data, "target_fs", FS))
            record_name = str(load_npz_scalar(data, "record_name", path.stem))
            subject_id = str(load_npz_scalar(data, "subject_id", ""))
            treadmill_type = str(load_npz_scalar(data, "treadmill_type", ""))
            snr_db = float(load_npz_scalar(data, "target_snr_db", math.nan))
            achieved_snr_db = float(load_npz_scalar(data, "achieved_snr_db", math.nan))

        LOGGER.info("Detector-repair baseline: %s at %g dB", record_name, snr_db)
        result = estimate_rpeaks_detector_repair(
            noisy_ecg,
            fs=fs,
            short_fraction=args.short_fraction,
            long_fraction=args.long_fraction,
        )
        rpeaks = np.asarray(result["rpeaks_samples"], dtype=np.int64)
        estimated_bpm, rr_counts = compute_bpm_windows(rpeaks, window_start_s, window_end_s, fs)
        metrics = compare_to_bpm0(estimated_bpm, bpm0)

        peak_path = output_dir / "peak_results" / f"{record_name}_snr_{snr_db:g}db_enhanced_peaks.npz".replace("-", "neg")
        save_peak_result(result, peak_path)

        row = {
            "record_name": record_name,
            "subject_id": subject_id,
            "treadmill_type": treadmill_type,
            "target_snr_db": snr_db,
            "achieved_snr_db": achieved_snr_db,
            "n_rpeaks": int(rpeaks.size),
            "n_ibi": int(max(0, rpeaks.size - 1)),
            "peak_result_path": str(peak_path),
            "n_xqrs_backbone_candidates": int(result["n_xqrs_backbone_candidates"]),
            "n_fill_candidates": int(result["n_fill_candidates"]),
            "n_pruned_from_xqrs": int(result["n_pruned_from_xqrs"]),
            "n_inserted_from_fill_candidates": int(result["n_inserted_from_fill_candidates"]),
            **metrics,
        }

        if not args.skip_beat_eval:
            if record_name not in reference_cache:
                reference_cache[record_name] = estimate_rpeaks_detector_repair(
                    clean_ecg,
                    fs=fs,
                    short_fraction=args.short_fraction,
                    long_fraction=args.long_fraction,
                )
            ref_peaks = np.asarray(reference_cache[record_name]["rpeaks_samples"], dtype=np.int64)
            match = match_rpeaks(ref_peaks, rpeaks, fs=fs)
            ibi_metrics = ibi_error_from_matched_peaks(ref_peaks, rpeaks, match, fs=fs)
            beat_row = {
                "record_name": record_name,
                "subject_id": subject_id,
                "treadmill_type": treadmill_type,
                "target_snr_db": snr_db,
                "reference_source": "derived_from_clean_ecg_detector_repair",
                "n_reference_rpeaks": int(ref_peaks.size),
                "n_estimated_rpeaks": int(rpeaks.size),
                "rpeak_tp": int(match["rpeak_tp"]),
                "rpeak_fp": int(match["rpeak_fp"]),
                "rpeak_fn": int(match["rpeak_fn"]),
                "rpeak_precision": float(match["rpeak_precision"]),
                "rpeak_recall": float(match["rpeak_recall"]),
                "rpeak_f1": float(match["rpeak_f1"]),
                "rpeak_mean_abs_timing_error_ms": float(match["rpeak_mean_abs_timing_error_ms"]),
                **ibi_metrics,
            }
            beat_rows.append(beat_row)
            row.update({k: v for k, v in beat_row.items() if k not in row})

        record_rows.append(row)

        for idx, (start_s, end_s, truth, est, rr_count) in enumerate(
            zip(window_start_s, window_end_s, bpm0, estimated_bpm, rr_counts, strict=True)
        ):
            window_rows.append(
                {
                    "record_name": record_name,
                    "subject_id": subject_id,
                    "treadmill_type": treadmill_type,
                    "target_snr_db": snr_db,
                    "achieved_snr_db": achieved_snr_db,
                    "window_index": idx,
                    "window_start_s": float(start_s),
                    "window_end_s": float(end_s),
                    "bpm0": float(truth),
                    "estimated_bpm": float(est),
                    "error_bpm": float(est - truth) if np.isfinite(est) and np.isfinite(truth) else math.nan,
                    "rr_count": int(rr_count),
                }
            )

        if not args.no_plots:
            label = f"snr_{snr_db:g}db".replace("-", "neg")
            plot_bpm_timeline(record_name, snr_db, window_start_s, bpm0, estimated_bpm, output_dir / "plots" / label / f"{record_name}_bpm_timeline.png")
            plot_example_peaks(record_name, snr_db, noisy_ecg, rpeaks, fs, output_dir / "plots" / label / f"{record_name}_example_peaks.png")

    by_snr = []
    for snr, rows in sorted(group_rows(window_rows, "target_snr_db").items(), key=lambda item: float(item[0])):
        by_snr.append(summarize_window_rows(rows, "target_snr_db", snr))
    overall = summarize_window_rows(window_rows, "group", "overall")
    beat_overall = summarize_beat_rows(beat_rows, "group", "overall") if beat_rows else {}

    write_csv(record_rows, output_dir / "metrics_by_record.csv")
    write_csv(window_rows, output_dir / "window_predictions.csv")
    write_csv(by_snr, output_dir / "metrics_by_snr.csv")
    write_csv(beat_rows, output_dir / "beat_level_metrics.csv")
    write_csv([beat_overall], output_dir / "beat_metrics_overall.csv")

    if not args.no_plots:
        plot_scatter(window_rows, output_dir / "plots" / "overall_pearson_scatter.png", "Detector-repair baseline Pearson scatter")
        plot_bland_altman(window_rows, output_dir / "plots" / "overall_bland_altman.png", "Detector-repair baseline Bland-Altman")

    summary = {
        "config": {
            "processed_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "short_fraction": float(args.short_fraction),
            "long_fraction": float(args.long_fraction),
            "skip_beat_eval": bool(args.skip_beat_eval),
        },
        "overall": overall,
        "by_snr": by_snr,
        "beat_overall": beat_overall,
        "n_records": len(record_rows),
        "n_windows": len(window_rows),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2)
    LOGGER.info("Detector-repair baseline evaluation complete: %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    hrv_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=hrv_dir / "processed_dataset")
    parser.add_argument("--output-dir", type=Path, default=hrv_dir / "grip_enhanced_results")
    parser.add_argument("--short-fraction", type=float, default=0.60)
    parser.add_argument("--long-fraction", type=float, default=1.80)
    parser.add_argument("--snrs", type=float, nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--skip-beat-eval", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    run_pipeline(args)


if __name__ == "__main__":
    main()
