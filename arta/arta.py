"""ARTA: ECG-only candidate-DAG + dynamic-programming IBI path.

1. Build a stronger ECG-only candidate set from WFDB-XQRS, Pan-Tompkins, and
   wavelet QRS detectors.
2. Merge close candidates with non-maximum suppression.
3. Construct a directed acyclic graph over candidate peaks, where edges are
   valid only for physiologic RR intervals.
4. Use dynamic programming with beam search to recover the minimum-cost beat
   path under local rhythm consistency, a time-varying HR prior, candidate
   quality, and coverage terms.

No PPG, accelerometer, or BPM0 labels are used by the detector.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import numpy as np

try:
    from .core_graph import (
        FS,
        backtrack_path,
        beam_search_grip,
        compare_to_bpm0,
        compute_bpm_windows,
        estimate_prior_hr,
        flag_ibi_outliers,
        group_rows,
        ibi_error_from_matched_peaks,
        json_safe,
        load_npz_scalar,
        match_rpeaks,
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
    )
    from .detectors import local_peak_quality, wfdb_xqrs_candidates
except ImportError:  # Allows direct execution from this directory.
    from core_graph import (
        FS,
        backtrack_path,
        beam_search_grip,
        compare_to_bpm0,
        compute_bpm_windows,
        estimate_prior_hr,
        flag_ibi_outliers,
        group_rows,
        ibi_error_from_matched_peaks,
        json_safe,
        load_npz_scalar,
        match_rpeaks,
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
    )
    from detectors import local_peak_quality, wfdb_xqrs_candidates

import json
import matplotlib

matplotlib.use("Agg")


LOGGER = logging.getLogger("arta")


def build_arta_candidate_pool(ecg: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create the ARTA candidate graph nodes.

    XQRS candidates are treated as high-confidence backbone candidates, while
    Pan-Tompkins and wavelet candidates add alternative nodes that can repair
    XQRS misses or help the DP skip noise-driven false positives.
    """
    xqrs_peaks = wfdb_xqrs_candidates(ecg, fs)
    xqrs_quality = local_peak_quality(ecg, xqrs_peaks, fs)
    xqrs_scores = np.clip(1.0 + 0.35 * xqrs_quality, 0.0, 1.35)

    pt_peaks, pt_scores = pan_tompkins_candidates(ecg, fs)
    wav_peaks, wav_scores = wavelet_candidates(ecg, fs)

    merged_peaks, merged_scores = merge_candidates_nms(
        [
            ("xqrs", xqrs_peaks, xqrs_scores),
            ("pan_tompkins", pt_peaks, 0.85 * np.asarray(pt_scores, dtype=np.float64)),
            ("wavelet", wav_peaks, 0.80 * np.asarray(wav_scores, dtype=np.float64)),
        ],
        ecg,
        fs,
        nms_ms=80.0,
    )

    counts = {
        "n_xqrs_candidates": int(xqrs_peaks.size),
        "n_pan_tompkins_candidates": int(pt_peaks.size),
        "n_wavelet_candidates": int(wav_peaks.size),
        "n_merged_candidates": int(merged_peaks.size),
    }
    return merged_peaks.astype(np.int64), merged_scores.astype(np.float32), counts


def estimate_rpeaks_arta(
    ecg: np.ndarray,
    fs: float = FS,
    beam_width: int = 3,
    sigma_hr_ms: float = 180.0,
    local_weight: float = 1.15,
    prior_weight: float = 0.75,
    candidate_weight: float = 0.18,
    beat_reward: float = 0.42,
    endpoint_weight: float = 0.04,
    coverage_weight: float = 0.25,
    omega_size: int = 5,
) -> dict[str, object]:
    """Estimate R-peaks using the ARTA DAG/DP shortest path.

    The selected path is the best beat trajectory through the candidate DAG.
    It is not allowed to use BPM0 labels; BPM0 is used only later for
    evaluation.
    """
    ecg = np.asarray(ecg, dtype=np.float64).squeeze()
    if ecg.ndim != 1:
        raise ValueError(f"Expected a 1D ECG signal, got shape {ecg.shape}")

    candidate_peaks, candidate_scores, counts = build_arta_candidate_pool(ecg, fs)
    prior_centers_s, prior_hr_bpm, prior_ibi_ms = estimate_prior_hr(candidate_peaks, ecg.size, fs)

    beams, best_node, best_state = beam_search_grip(
        candidate_peaks,
        candidate_scores,
        prior_centers_s,
        prior_ibi_ms,
        n_samples=ecg.size,
        fs=fs,
        beam_width=beam_width,
        omega_size=omega_size,
        sigma_hr_ms=sigma_hr_ms,
        local_weight=local_weight,
        prior_weight=prior_weight,
        candidate_weight=candidate_weight,
        beat_reward=beat_reward,
        endpoint_weight=endpoint_weight,
        coverage_weight=coverage_weight,
    )
    rpeaks = backtrack_path(candidate_peaks, beams, best_node, best_state)

    ibi_ms = np.diff(rpeaks) / float(fs) * 1000.0
    instant_bpm = 60000.0 / ibi_ms if ibi_ms.size else np.array([], dtype=np.float64)
    outlier_mask = flag_ibi_outliers(ibi_ms)

    return {
        "rpeaks_samples": rpeaks.astype(np.int64),
        "rpeaks_seconds": (rpeaks / float(fs)).astype(np.float32),
        "ibi_ms": ibi_ms.astype(np.float32),
        "instant_bpm": instant_bpm.astype(np.float32),
        "ibi_outlier_mask": outlier_mask,
        "candidate_peaks": candidate_peaks.astype(np.int64),
        "candidate_scores": candidate_scores.astype(np.float32),
        "prior_centers_s": prior_centers_s,
        "prior_hr_bpm": prior_hr_bpm,
        "prior_ibi_ms": prior_ibi_ms,
        **counts,
        "beam_width": int(beam_width),
        "omega_size": int(omega_size),
        "sigma_hr_ms": float(sigma_hr_ms),
        "local_weight": float(local_weight),
        "prior_weight": float(prior_weight),
        "candidate_weight": float(candidate_weight),
        "beat_reward": float(beat_reward),
        "endpoint_weight": float(endpoint_weight),
        "coverage_weight": float(coverage_weight),
        "best_node": -1 if best_node is None else int(best_node),
        "best_state": -1 if best_state is None else int(best_state),
        "n_beam_nodes": int(len(beams)),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    """Run ARTA alone and save the same reports as the other runners."""
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

        LOGGER.info("ARTA DAG+DP: %s at %g dB", record_name, snr_db)
        result = estimate_rpeaks_arta(
            noisy_ecg,
            fs=fs,
            beam_width=args.beam_width,
            sigma_hr_ms=args.sigma_hr_ms,
            omega_size=args.omega_size,
        )
        rpeaks = np.asarray(result["rpeaks_samples"], dtype=np.int64)
        estimated_bpm, rr_counts = compute_bpm_windows(rpeaks, window_start_s, window_end_s, fs)
        metrics = compare_to_bpm0(estimated_bpm, bpm0)

        peak_path = output_dir / "peak_results" / f"{record_name}_snr_{snr_db:g}db_arta_peaks.npz".replace("-", "neg")
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
            **{k: v for k, v in result.items() if isinstance(v, (int, float, str, bool, np.integer, np.floating))},
            **metrics,
        }
        record_rows.append(row)

        if record_name not in reference_cache:
            reference_cache[record_name] = estimate_rpeaks_arta(
                clean_ecg,
                fs=fs,
                beam_width=args.beam_width,
                sigma_hr_ms=args.sigma_hr_ms,
                omega_size=args.omega_size,
            )
        ref_peaks = np.asarray(reference_cache[record_name]["rpeaks_samples"], dtype=np.int64)
        match = match_rpeaks(ref_peaks, rpeaks, fs=fs)
        beat_row = {
            "record_name": record_name,
            "subject_id": subject_id,
            "treadmill_type": treadmill_type,
            "target_snr_db": snr_db,
            "reference_source": "derived_from_clean_ecg_arta",
            "n_reference_rpeaks": int(ref_peaks.size),
            "n_estimated_rpeaks": int(rpeaks.size),
            **match,
            **ibi_error_from_matched_peaks(ref_peaks, rpeaks, match, fs=fs),
        }
        beat_rows.append(beat_row)
        row.update({k: v for k, v in beat_row.items() if k not in row})

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
            plot_bpm_timeline(
                window_start_s,
                window_end_s,
                bpm0,
                estimated_bpm,
                output_dir / "plots" / "bpm_timeline" / f"{record_name}_snr_{snr_db:g}db.png".replace("-", "neg"),
                title=f"ARTA BPM: {record_name}, SNR {snr_db:g} dB",
            )
            plot_example_peaks(
                noisy_ecg,
                rpeaks,
                fs,
                output_dir / "plots" / "example_peaks" / f"{record_name}_snr_{snr_db:g}db.png".replace("-", "neg"),
                title=f"ARTA R-peaks: {record_name}, SNR {snr_db:g} dB",
            )

    by_snr = []
    for snr, rows in sorted(group_rows(window_rows, "target_snr_db").items(), key=lambda item: float(item[0])):
        by_snr.append(summarize_window_rows(rows, "target_snr_db", snr))
    by_record = [summarize_window_rows(rows, "record_name", record) for record, rows in group_rows(window_rows, "record_name").items()]
    by_type = [summarize_window_rows(rows, "treadmill_type", typ) for typ, rows in group_rows(window_rows, "treadmill_type").items()]
    beat_by_snr = [summarize_beat_rows(rows, "target_snr_db", snr) for snr, rows in group_rows(beat_rows, "target_snr_db").items()]

    write_csv(record_rows, output_dir / "metrics_by_record.csv")
    write_csv(window_rows, output_dir / "window_predictions.csv")
    write_csv(by_snr, output_dir / "metrics_by_snr.csv")
    write_csv(by_record, output_dir / "metrics_by_record_summary.csv")
    write_csv(by_type, output_dir / "metrics_by_treadmill_type.csv")
    write_csv(beat_rows, output_dir / "beat_level_metrics.csv")
    write_csv(beat_by_snr, output_dir / "beat_metrics_by_snr.csv")

    if not args.no_plots:
        plot_scatter(window_rows, output_dir / "plots" / "overall_pearson_scatter.png", "ARTA Pearson scatter")
        plot_bland_altman(window_rows, output_dir / "plots" / "overall_bland_altman.png", "ARTA Bland-Altman")

    summary = {
        "config": {
            "processed_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "beam_width": int(args.beam_width),
            "omega_size": int(args.omega_size),
            "sigma_hr_ms": float(args.sigma_hr_ms),
        },
        "metrics_by_snr": by_snr,
        "metrics_by_record": by_record,
        "metrics_by_treadmill_type": by_type,
        "beat_metrics_by_snr": beat_by_snr,
        "overall": summarize_window_rows(window_rows, "overall", "all"),
        "n_records": len(record_rows),
        "n_windows": len(window_rows),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2)
    LOGGER.info("ARTA evaluation complete: %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    hrv_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=hrv_dir / "processed_dataset")
    parser.add_argument("--output-dir", type=Path, default=hrv_dir / "arta_results")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--omega-size", type=int, default=5)
    parser.add_argument("--sigma-hr-ms", type=float, default=180.0)
    parser.add_argument("--snrs", type=float, nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    run_pipeline(args)


if __name__ == "__main__":
    main()
