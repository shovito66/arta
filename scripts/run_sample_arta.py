"""Run ARTA on the included compact sample dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arta.core_graph import compare_to_bpm0, compute_bpm_windows, load_npz_scalar
from arta.arta import estimate_rpeaks_arta


def run_one(path: Path, output_dir: Path, beam_width: int, omega_size: int, sigma_hr_ms: float) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        ecg = np.asarray(data["noisy_ecg"] if "noisy_ecg" in data else data["clean_ecg"], dtype=np.float32)
        fs = float(load_npz_scalar(data, "target_fs", 500.0))
        bpm0 = np.asarray(data["bpm0"], dtype=np.float32)
        window_start_s = np.asarray(data["window_start_s"], dtype=np.float32)
        window_end_s = np.asarray(data["window_end_s"], dtype=np.float32)
        record_name = str(load_npz_scalar(data, "record_name", path.stem))
        snr_db = float(load_npz_scalar(data, "target_snr_db", np.nan))

    result = estimate_rpeaks_arta(
        ecg,
        fs=fs,
        beam_width=beam_width,
        omega_size=omega_size,
        sigma_hr_ms=sigma_hr_ms,
    )
    estimated_bpm, rr_counts = compute_bpm_windows(result["rpeaks_samples"], window_start_s, window_end_s, fs)
    metrics = compare_to_bpm0(estimated_bpm, bpm0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir / f"{record_name}_snr_{snr_db:g}db_arta_peaks.npz".replace("-", "neg")
    np.savez_compressed(
        out_npz,
        rpeaks_samples=np.asarray(result["rpeaks_samples"], dtype=np.int64),
        rpeaks_seconds=np.asarray(result["rpeaks_seconds"], dtype=np.float32),
        ibi_ms=np.asarray(result["ibi_ms"], dtype=np.float32),
        estimated_bpm=np.asarray(estimated_bpm, dtype=np.float32),
        rr_counts=np.asarray(rr_counts, dtype=np.int64),
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        bpm0=bpm0,
    )

    return {
        "record_name": record_name,
        "snr_db": None if not np.isfinite(snr_db) else snr_db,
        "n_rpeaks": int(len(result["rpeaks_samples"])),
        "peak_file": str(out_npz),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "sample_data" / "processed_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sample_arta")
    parser.add_argument("--snr", type=float, default=-2.0)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--omega-size", type=int, default=5)
    parser.add_argument("--sigma-hr-ms", type=float, default=180.0)
    args = parser.parse_args()

    snr_folder = f"snr_{args.snr:g}db".replace("-", "neg")
    files = sorted((args.processed_dir / "noisy_records" / snr_folder).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No sample files found under {args.processed_dir / 'noisy_records' / snr_folder}")

    rows = [run_one(path, args.output_dir, args.beam_width, args.omega_size, args.sigma_hr_ms) for path in files]
    summary_path = args.output_dir / "sample_metrics.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
