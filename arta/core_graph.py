"""Run the ECG-only hybrid GRIP-IBI algorithm on prepared noisy ECG data.

This script consumes the Step-1 processed dataset:

    HRV/processed_dataset/noisy_records/snr_*db/*.npz

It estimates R-peaks, IBIs, and 8-second-window BPM from noisy ECG only, then
compares estimated BPM against the BPM0 ground-truth traces. It also derives a
clean-ECG reference peak set for optional beat-level diagnostics because the
treadmill files do not include manual beat annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, fftconvolve, filtfilt, find_peaks, savgol_filter


FS = 500.0
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
WINDOW_SEC = 8.0
WINDOW_STEP_SEC = 2.0
LOGGER = logging.getLogger("grip_ibi")


@dataclass(frozen=True)
class BeamState:
    """One partial GRIP path ending at a candidate node."""

    cost: float
    parent_node: int | None
    parent_state: int | None
    recent_ibi_ms: tuple[float, ...]
    path_length: int
    first_node: int


def _as_1d_float(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D ECG signal, got shape {x.shape}")
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Signal contains no finite samples")
    if not finite.all():
        idx = np.arange(x.size)
        x = x.copy()
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Butterworth bandpass with short-signal safeguards."""
    x = _as_1d_float(signal)
    if x.size < max(16, order * 6):
        return x - np.median(x)

    nyq = 0.5 * float(fs)
    low = max(float(low), 0.001)
    high = min(float(high), nyq * 0.95)
    if low >= high:
        return x - np.median(x)

    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    padlen = min(3 * (max(len(a), len(b)) - 1), x.size - 1)
    return filtfilt(b, a, x, padlen=padlen)


def moving_average(x: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    return np.convolve(x, np.ones(samples, dtype=np.float64) / samples, mode="same")


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = np.percentile(values, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(values))
        lo = float(np.min(values))
    if hi <= lo:
        return np.ones(values.size, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def refine_to_local_extreme(peaks: Iterable[int], score_signal: np.ndarray, fs: float, radius_ms: float = 80.0) -> np.ndarray:
    """Move rough peak locations to the strongest local ECG deflection."""
    radius = max(1, int(round(radius_ms / 1000.0 * fs)))
    refined: list[int] = []
    score_signal = np.asarray(score_signal)
    for peak in peaks:
        peak = int(peak)
        start = max(0, peak - radius)
        stop = min(score_signal.size, peak + radius + 1)
        if stop > start:
            refined.append(start + int(np.argmax(score_signal[start:stop])))
    return np.asarray(refined, dtype=np.int64)


def pan_tompkins_candidates(ecg: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Generate R-peak candidates with a Pan-Tompkins-style detector."""
    x = _as_1d_float(ecg)
    qrs_band = bandpass_filter(x, fs, 5.0, 18.0, order=2)
    broad_band = bandpass_filter(x, fs, 0.5, 40.0, order=3)

    derivative = np.gradient(qrs_band)
    energy = derivative**2
    integrated = moving_average(energy, int(round(0.150 * fs)))

    min_distance = int(round(RR_MIN_MS / 1000.0 * fs))
    height = np.percentile(integrated, 65)
    prominence = max(np.std(integrated) * 0.25, np.finfo(float).eps)
    rough_peaks, props = find_peaks(
        integrated,
        distance=min_distance,
        height=height,
        prominence=prominence,
    )

    if rough_peaks.size < 2:
        rough_peaks, props = find_peaks(
            integrated,
            distance=min_distance,
            height=np.percentile(integrated, 50),
        )

    score_signal = np.abs(broad_band)
    refined = refine_to_local_extreme(rough_peaks, score_signal, fs)
    refined = np.unique(refined)
    raw_scores = score_signal[refined] if refined.size else np.array([], dtype=float)
    return refined, normalize_scores(raw_scores)


def ricker_wavelet(points: int, width: float) -> np.ndarray:
    """Small Mexican-hat/Ricker wavelet implementation for QRS-scale CWT."""
    points = int(points)
    if points % 2 == 0:
        points += 1
    half = points // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    wavelet = (1.0 - (x / width) ** 2) * np.exp(-(x**2) / (2.0 * width**2))
    wavelet -= np.mean(wavelet)
    norm = np.linalg.norm(wavelet)
    if norm > 0:
        wavelet /= norm
    return wavelet


def wavelet_candidates(ecg: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Generate R-peak candidates using a Ricker wavelet response envelope."""
    x = _as_1d_float(ecg)
    qrs_band = bandpass_filter(x, fs, 3.0, 30.0, order=2)
    broad_band = bandpass_filter(x, fs, 0.5, 40.0, order=3)

    widths = [max(4, int(round(ms / 1000.0 * fs))) for ms in (25.0, 40.0, 60.0)]
    responses = []
    for width in widths:
        kernel = ricker_wavelet(points=width * 8 + 1, width=float(width))
        responses.append(np.abs(fftconvolve(qrs_band, kernel[::-1], mode="same")))
    envelope = np.max(np.vstack(responses), axis=0)
    envelope = moving_average(envelope, int(round(0.040 * fs)))

    min_distance = int(round(RR_MIN_MS / 1000.0 * fs))
    rough_peaks, _ = find_peaks(
        envelope,
        distance=min_distance,
        height=np.percentile(envelope, 65),
        prominence=max(np.std(envelope) * 0.20, np.finfo(float).eps),
    )
    if rough_peaks.size < 2:
        rough_peaks, _ = find_peaks(envelope, distance=min_distance, height=np.percentile(envelope, 50))

    score_signal = np.abs(broad_band)
    refined = refine_to_local_extreme(rough_peaks, score_signal, fs)
    refined = np.unique(refined)
    raw_scores = score_signal[refined] + envelope[refined] if refined.size else np.array([], dtype=float)
    return refined, normalize_scores(raw_scores)


def merge_candidates_nms(
    candidate_sets: list[tuple[str, np.ndarray, np.ndarray]],
    ecg: np.ndarray,
    fs: float = FS,
    nms_ms: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge detector candidate sets with deterministic non-maximum suppression."""
    all_items: list[tuple[int, float, str]] = []
    for label, peaks, scores in candidate_sets:
        for peak, score in zip(peaks, scores, strict=False):
            all_items.append((int(peak), float(score), label))

    if not all_items:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    all_items.sort(key=lambda item: (item[0], -item[1], item[2]))
    score_signal = np.abs(bandpass_filter(ecg, fs, 0.5, 40.0, order=3))
    radius = int(round(nms_ms / 1000.0 * fs))

    merged_peaks: list[int] = []
    merged_scores: list[float] = []
    cluster: list[tuple[int, float, str]] = []

    def flush_cluster(items: list[tuple[int, float, str]]) -> None:
        if not items:
            return
        unique_sources = {item[2] for item in items}
        best = min(
            items,
            key=lambda item: (
                -item[1],
                -float(score_signal[min(max(item[0], 0), score_signal.size - 1)]),
                item[0],
            ),
        )
        best_peak = refine_to_local_extreme([best[0]], score_signal, fs, radius_ms=40.0)[0]
        source_bonus = 0.20 * max(0, len(unique_sources) - 1)
        merged_peaks.append(int(best_peak))
        merged_scores.append(float(best[1]) + source_bonus)

    for item in all_items:
        if not cluster or item[0] - cluster[-1][0] <= radius:
            cluster.append(item)
        else:
            flush_cluster(cluster)
            cluster = [item]
    flush_cluster(cluster)

    peaks = np.asarray(merged_peaks, dtype=np.int64)
    scores = normalize_scores(np.asarray(merged_scores, dtype=np.float64))
    order = np.argsort(peaks, kind="mergesort")
    return peaks[order], scores[order]


def estimate_prior_hr(
    rough_peaks: np.ndarray,
    n_samples: int,
    fs: float = FS,
    window_sec: float = WINDOW_SEC,
    step_sec: float = WINDOW_STEP_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a slowly varying HR prior from ECG-derived rough peaks."""
    rough_peaks = np.asarray(rough_peaks, dtype=np.int64)
    window_samples = int(round(window_sec * fs))
    step_samples = int(round(step_sec * fs))
    starts = np.arange(0, max(1, n_samples - window_samples + 1), step_samples, dtype=np.int64)
    centers_s = (starts + window_samples / 2.0) / float(fs)
    prior_hr = np.full(starts.size, np.nan, dtype=np.float64)

    for idx, start in enumerate(starts):
        end = start + window_samples
        local = rough_peaks[(rough_peaks >= start) & (rough_peaks < end)]
        if local.size >= 2:
            rr_ms = np.diff(local) / float(fs) * 1000.0
            rr_ms = rr_ms[(rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]
            if rr_ms.size:
                prior_hr[idx] = 60000.0 / np.median(rr_ms)
        elif local.size == 1:
            prior_hr[idx] = 60.0 / window_sec

    valid = np.isfinite(prior_hr)
    if not valid.any():
        prior_hr[:] = 75.0
    elif not valid.all():
        prior_hr[~valid] = np.interp(centers_s[~valid], centers_s[valid], prior_hr[valid])

    if prior_hr.size >= 7:
        win = min(prior_hr.size if prior_hr.size % 2 == 1 else prior_hr.size - 1, 11)
        if win >= 5:
            prior_hr = savgol_filter(prior_hr, window_length=win, polyorder=2, mode="interp")

    prior_hr = np.clip(prior_hr, 30.0, 220.0)
    prior_ibi_ms = 60000.0 / prior_hr
    return centers_s.astype(np.float32), prior_hr.astype(np.float32), prior_ibi_ms.astype(np.float32)


def build_dag(
    candidate_peaks: np.ndarray,
    fs: float = FS,
    rr_min_ms: float = RR_MIN_MS,
    rr_max_ms: float = RR_MAX_MS,
) -> list[np.ndarray]:
    """Build predecessor lists for a DAG over physiologically valid RR edges."""
    peaks = np.asarray(candidate_peaks, dtype=np.int64)
    rr_min = int(round(rr_min_ms / 1000.0 * fs))
    rr_max = int(round(rr_max_ms / 1000.0 * fs))
    predecessors: list[np.ndarray] = []
    for i, peak in enumerate(peaks):
        left = np.searchsorted(peaks, peak - rr_max, side="left")
        right = np.searchsorted(peaks, peak - rr_min, side="right")
        predecessors.append(np.arange(left, min(right, i), dtype=np.int64))
    return predecessors


def local_rhythm_cost(rr_ms: float, recent_ibi_ms: tuple[float, ...], epsilon_ms: float) -> float:
    """Median/MAD rhythm consistency cost for one new interval."""
    if not recent_ibi_ms:
        return 0.0
    recent = np.asarray(recent_ibi_ms, dtype=np.float64)
    median = float(np.median(recent))
    mad = float(np.median(np.abs(recent - median)))
    return abs(float(rr_ms) - median) / (mad + epsilon_ms)


def beam_search_grip(
    candidate_peaks: np.ndarray,
    candidate_scores: np.ndarray,
    prior_centers_s: np.ndarray,
    prior_ibi_ms: np.ndarray,
    n_samples: int,
    fs: float = FS,
    beam_width: int = 3,
    omega_size: int = 5,
    sigma_hr_ms: float = 200.0,
    epsilon_ms: float = 30.0,
    local_weight: float = 1.0,
    prior_weight: float = 0.8,
    candidate_weight: float = 0.25,
    beat_reward: float = 0.35,
    endpoint_weight: float = 0.04,
    coverage_weight: float = 0.25,
    start_gap_weight: float = 0.25,
) -> tuple[list[list[BeamState]], int | None, int | None]:
    """Dynamic programming with beam search for the GRIP peak path.

    Each candidate peak is a DAG node. Edges are valid RR intervals only.
    For every node we keep the top B partial paths. Transition costs combine
    a local robust rhythm model from recent IBIs (median/MAD) and a soft,
    time-varying HR prior estimated from the same noisy ECG.
    """
    peaks = np.asarray(candidate_peaks, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if peaks.size == 0:
        return [], None, None

    predecessors = build_dag(peaks, fs=fs)
    beams: list[list[BeamState]] = []

    for i, peak in enumerate(peaks):
        t_s = peak / float(fs)
        prior_ibi = float(np.interp(t_s, prior_centers_s, prior_ibi_ms))
        node_quality_cost = candidate_weight * (1.0 - float(scores[i]))
        initial_start_cost = start_gap_weight * t_s

        states: list[BeamState] = [
            BeamState(
                cost=node_quality_cost + initial_start_cost,
                parent_node=None,
                parent_state=None,
                recent_ibi_ms=(),
                path_length=1,
                first_node=i,
            )
        ]

        for pred in predecessors[i]:
            rr_ms = (peak - peaks[pred]) / float(fs) * 1000.0
            prior_cost = abs(rr_ms - prior_ibi) / sigma_hr_ms
            for pred_state_idx, pred_state in enumerate(beams[pred]):
                rhythm_cost = local_rhythm_cost(rr_ms, pred_state.recent_ibi_ms, epsilon_ms)
                cost = (
                    pred_state.cost
                    + local_weight * rhythm_cost
                    + prior_weight * prior_cost
                    + node_quality_cost
                    - beat_reward
                )
                recent = (pred_state.recent_ibi_ms + (float(rr_ms),))[-omega_size:]
                states.append(
                    BeamState(
                        cost=float(cost),
                        parent_node=int(pred),
                        parent_state=int(pred_state_idx),
                        recent_ibi_ms=recent,
                        path_length=pred_state.path_length + 1,
                        first_node=pred_state.first_node,
                    )
                )

        # Rank by average path cost, not raw accumulated cost. Raw cost naturally
        # favors short paths because they contain fewer transitions; GRIP needs
        # full beat trajectories, so the beam preserves low average-cost paths
        # while using path length as a deterministic tie-breaker.
        states.sort(
            key=lambda state: (
                round(state.cost / max(1, state.path_length - 1), 12),
                -state.path_length,
                peaks[state.first_node],
                -float(scores[i]),
                state.parent_node if state.parent_node is not None else -1,
                state.parent_state if state.parent_state is not None else -1,
            )
        )
        beams.append(states[: max(1, int(beam_width))])

    duration_s = n_samples / float(fs)
    best_key: tuple[float, int, int, int] | None = None
    best_node: int | None = None
    best_state: int | None = None
    for node_idx, node_states in enumerate(beams):
        last_t = peaks[node_idx] / float(fs)
        for state_idx, state in enumerate(node_states):
            first_t = peaks[state.first_node] / float(fs)
            coverage_s = max(0.0, last_t - first_t)
            uncovered_s = max(0.0, duration_s * 0.95 - coverage_s)
            endpoint_penalty = endpoint_weight * (first_t + max(0.0, duration_s - last_t)) + coverage_weight * uncovered_s
            average_cost = state.cost / max(1, state.path_length - 1)
            final_cost = average_cost + endpoint_penalty
            key = (round(final_cost, 12), -state.path_length, peaks[state.first_node], peaks[node_idx])
            if best_key is None or key < best_key:
                best_key = key
                best_node = node_idx
                best_state = state_idx

    return beams, best_node, best_state


def backtrack_path(candidate_peaks: np.ndarray, beams: list[list[BeamState]], best_node: int | None, best_state: int | None) -> np.ndarray:
    """Recover the best R-peak path from beam-search parent pointers."""
    if best_node is None or best_state is None:
        return np.array([], dtype=np.int64)

    path_nodes: list[int] = []
    node = best_node
    state_idx = best_state
    while node is not None and state_idx is not None:
        path_nodes.append(node)
        state = beams[node][state_idx]
        node = state.parent_node
        state_idx = state.parent_state

    path_nodes.reverse()
    return np.asarray(candidate_peaks[path_nodes], dtype=np.int64)


def flag_ibi_outliers(ibi_ms: np.ndarray, local_beats: int = 5, threshold_frac: float = 0.20) -> np.ndarray:
    """Flag IBIs that deviate more than threshold_frac from a local median."""
    ibi_ms = np.asarray(ibi_ms, dtype=np.float64)
    flags = np.zeros(ibi_ms.size, dtype=bool)
    if ibi_ms.size < 3:
        return flags
    half = max(1, local_beats // 2)
    for idx, value in enumerate(ibi_ms):
        start = max(0, idx - half)
        stop = min(ibi_ms.size, idx + half + 1)
        local = ibi_ms[start:stop]
        median = float(np.median(local))
        if median > 0 and abs(value - median) > threshold_frac * median:
            flags[idx] = True
    return flags


def estimate_rpeaks_grip(
    ecg: np.ndarray,
    fs: float = FS,
    beam_width: int = 3,
    sigma_hr_ms: float = 200.0,
) -> dict[str, np.ndarray | float | int]:
    """Full ECG-only hybrid GRIP-IBI R-peak estimator."""
    ecg = _as_1d_float(ecg)
    pt_peaks, pt_scores = pan_tompkins_candidates(ecg, fs)
    wav_peaks, wav_scores = wavelet_candidates(ecg, fs)
    merged_peaks, merged_scores = merge_candidates_nms(
        [
            ("pan_tompkins", pt_peaks, pt_scores),
            ("wavelet", wav_peaks, wav_scores),
        ],
        ecg,
        fs,
    )

    prior_centers_s, prior_hr_bpm, prior_ibi_ms = estimate_prior_hr(merged_peaks, ecg.size, fs)
    beams, best_node, best_state = beam_search_grip(
        merged_peaks,
        merged_scores,
        prior_centers_s,
        prior_ibi_ms,
        n_samples=ecg.size,
        fs=fs,
        beam_width=beam_width,
        sigma_hr_ms=sigma_hr_ms,
    )
    rpeaks = backtrack_path(merged_peaks, beams, best_node, best_state)
    ibi_ms = np.diff(rpeaks) / float(fs) * 1000.0
    bpm = 60000.0 / ibi_ms if ibi_ms.size else np.array([], dtype=np.float64)
    outlier_mask = flag_ibi_outliers(ibi_ms)

    return {
        "rpeaks_samples": rpeaks.astype(np.int64),
        "rpeaks_seconds": (rpeaks / float(fs)).astype(np.float32),
        "ibi_ms": ibi_ms.astype(np.float32),
        "instant_bpm": bpm.astype(np.float32),
        "ibi_outlier_mask": outlier_mask,
        "candidate_peaks": merged_peaks.astype(np.int64),
        "candidate_scores": merged_scores.astype(np.float32),
        "prior_centers_s": prior_centers_s,
        "prior_hr_bpm": prior_hr_bpm,
        "prior_ibi_ms": prior_ibi_ms,
        "n_pan_tompkins_candidates": int(pt_peaks.size),
        "n_wavelet_candidates": int(wav_peaks.size),
        "n_merged_candidates": int(merged_peaks.size),
        "beam_width": int(beam_width),
        "sigma_hr_ms": float(sigma_hr_ms),
    }


def compute_bpm_windows(
    rpeaks_samples: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    fs: float = FS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert estimated R-peaks to BPM values in BPM0-aligned windows."""
    rpeaks = np.asarray(rpeaks_samples, dtype=np.int64)
    starts = np.asarray(window_start_s, dtype=np.float64)
    ends = np.asarray(window_end_s, dtype=np.float64)
    estimated_bpm = np.full(starts.size, np.nan, dtype=np.float64)
    rr_counts = np.zeros(starts.size, dtype=np.int64)

    if rpeaks.size >= 2:
        rr_ms = np.diff(rpeaks) / float(fs) * 1000.0
        rr_mid_s = (rpeaks[:-1] + rpeaks[1:]) / (2.0 * float(fs))
        inst_bpm = 60000.0 / rr_ms
    else:
        rr_ms = np.array([], dtype=np.float64)
        rr_mid_s = np.array([], dtype=np.float64)
        inst_bpm = np.array([], dtype=np.float64)

    for idx, (start_s, end_s) in enumerate(zip(starts, ends, strict=True)):
        mask = (rr_mid_s >= start_s) & (rr_mid_s < end_s)
        if np.any(mask):
            estimated_bpm[idx] = float(np.mean(inst_bpm[mask]))
            rr_counts[idx] = int(np.sum(mask))
        else:
            local_peaks = rpeaks[(rpeaks / float(fs) >= start_s) & (rpeaks / float(fs) < end_s)]
            duration_s = max(end_s - start_s, np.finfo(float).eps)
            if local_peaks.size > 0:
                estimated_bpm[idx] = float(local_peaks.size / duration_s * 60.0)
                rr_counts[idx] = max(0, int(local_peaks.size - 1))

    return estimated_bpm.astype(np.float32), rr_counts


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return math.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def bland_altman(estimated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.isfinite(estimated) & np.isfinite(truth)
    if not np.any(mask):
        return {"bias": math.nan, "loa_lower": math.nan, "loa_upper": math.nan, "sd_diff": math.nan}
    diff = estimated[mask] - truth[mask]
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return {
        "bias": bias,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "sd_diff": sd,
    }


def compare_to_bpm0(estimated_bpm: np.ndarray, bpm0: np.ndarray) -> dict[str, float | int]:
    """Compute window-level BPM metrics against BPM0."""
    estimated = np.asarray(estimated_bpm, dtype=np.float64)
    truth = np.asarray(bpm0, dtype=np.float64)
    mask = np.isfinite(estimated) & np.isfinite(truth)
    n = int(np.sum(mask))
    if n == 0:
        return {
            "n_windows": 0,
            "pearson_r": math.nan,
            "mae_bpm": math.nan,
            "rmse_bpm": math.nan,
            "within_5_bpm_pct": math.nan,
            **bland_altman(estimated, truth),
        }

    errors = estimated[mask] - truth[mask]
    metrics: dict[str, float | int] = {
        "n_windows": n,
        "pearson_r": pearson_corr(estimated, truth),
        "mae_bpm": float(np.mean(np.abs(errors))),
        "rmse_bpm": float(np.sqrt(np.mean(errors**2))),
        "within_5_bpm_pct": float(np.mean(np.abs(errors) <= 5.0) * 100.0),
    }
    metrics.update(bland_altman(estimated, truth))
    return metrics


def match_rpeaks(reference: np.ndarray, estimated: np.ndarray, tolerance_ms: float = 100.0, fs: float = FS) -> dict[str, object]:
    """Greedy one-to-one peak matching for derived beat-level diagnostics."""
    ref = np.asarray(reference, dtype=np.int64)
    est = np.asarray(estimated, dtype=np.int64)
    tol = int(round(tolerance_ms / 1000.0 * fs))
    matched_ref: list[int] = []
    matched_est: list[int] = []
    timing_errors_ms: list[float] = []
    est_used = np.zeros(est.size, dtype=bool)

    for ref_idx, ref_peak in enumerate(ref):
        left = np.searchsorted(est, ref_peak - tol, side="left")
        right = np.searchsorted(est, ref_peak + tol, side="right")
        candidates = [idx for idx in range(left, right) if not est_used[idx]]
        if not candidates:
            continue
        best_est_idx = min(candidates, key=lambda idx: (abs(est[idx] - ref_peak), est[idx]))
        est_used[best_est_idx] = True
        matched_ref.append(ref_idx)
        matched_est.append(best_est_idx)
        timing_errors_ms.append((est[best_est_idx] - ref_peak) / float(fs) * 1000.0)

    tp = len(matched_ref)
    fp = int(est.size - tp)
    fn = int(ref.size - tp)
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan

    return {
        "matched_ref_indices": np.asarray(matched_ref, dtype=np.int64),
        "matched_est_indices": np.asarray(matched_est, dtype=np.int64),
        "timing_errors_ms": np.asarray(timing_errors_ms, dtype=np.float32),
        "rpeak_tp": tp,
        "rpeak_fp": fp,
        "rpeak_fn": fn,
        "rpeak_precision": float(precision),
        "rpeak_recall": float(recall),
        "rpeak_f1": float(f1),
        "rpeak_mean_abs_timing_error_ms": float(np.mean(np.abs(timing_errors_ms))) if timing_errors_ms else math.nan,
    }


def ibi_error_from_matched_peaks(reference: np.ndarray, estimated: np.ndarray, match: dict[str, object], fs: float = FS) -> dict[str, float | int]:
    """Compute IBI error where consecutive reference beats both match estimated beats."""
    ref = np.asarray(reference, dtype=np.int64)
    est = np.asarray(estimated, dtype=np.int64)
    matched_ref = np.asarray(match["matched_ref_indices"], dtype=np.int64)
    matched_est = np.asarray(match["matched_est_indices"], dtype=np.int64)
    ref_to_est = {int(r): int(e) for r, e in zip(matched_ref, matched_est, strict=False)}

    errors: list[float] = []
    for ref_idx in range(ref.size - 1):
        if ref_idx in ref_to_est and ref_idx + 1 in ref_to_est:
            ref_ibi = (ref[ref_idx + 1] - ref[ref_idx]) / float(fs) * 1000.0
            est_i = ref_to_est[ref_idx]
            est_j = ref_to_est[ref_idx + 1]
            est_ibi = (est[est_j] - est[est_i]) / float(fs) * 1000.0
            errors.append(est_ibi - ref_ibi)

    err = np.asarray(errors, dtype=np.float64)
    if err.size == 0:
        return {
            "n_ibi_matched": 0,
            "ibi_error_mean_ms": math.nan,
            "ibi_error_mae_ms": math.nan,
            "ibi_error_rmse_ms": math.nan,
            "ibi_error_sd_ms": math.nan,
        }
    return {
        "n_ibi_matched": int(err.size),
        "ibi_error_mean_ms": float(np.mean(err)),
        "ibi_error_mae_ms": float(np.mean(np.abs(err))),
        "ibi_error_rmse_ms": float(np.sqrt(np.mean(err**2))),
        "ibi_error_sd_ms": float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
    }


def plot_bpm_timeline(record_name: str, snr_db: float, window_start_s: np.ndarray, bpm0: np.ndarray, estimated_bpm: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(window_start_s, bpm0, label="BPM0 ground truth", linewidth=1.4)
    plt.plot(window_start_s, estimated_bpm, label="GRIP estimated BPM", linewidth=1.2)
    plt.title(f"{record_name} BPM timeline ({snr_db:g} dB)")
    plt.xlabel("Window start time (s)")
    plt.ylabel("BPM")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_example_peaks(record_name: str, snr_db: float, ecg: np.ndarray, rpeaks: np.ndarray, fs: float, out_path: Path, seconds: float = 12.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_samples = min(ecg.size, int(round(seconds * fs)))
    local_peaks = rpeaks[rpeaks < max_samples]
    time = np.arange(max_samples) / float(fs)
    plt.figure(figsize=(12, 4))
    plt.plot(time, ecg[:max_samples], linewidth=0.7, label="Noisy ECG")
    if local_peaks.size:
        plt.scatter(local_peaks / float(fs), ecg[local_peaks], s=18, color="crimson", label="GRIP R-peaks", zorder=3)
    plt.title(f"{record_name} detected R-peaks ({snr_db:g} dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth = np.asarray([row["bpm0"] for row in rows], dtype=float)
    est = np.asarray([row["estimated_bpm"] for row in rows], dtype=float)
    mask = np.isfinite(truth) & np.isfinite(est)
    plt.figure(figsize=(5, 5))
    plt.scatter(truth[mask], est[mask], s=8, alpha=0.45)
    if np.any(mask):
        lo = float(min(np.min(truth[mask]), np.min(est[mask])))
        hi = float(max(np.max(truth[mask]), np.max(est[mask])))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("BPM0 ground truth")
    plt.ylabel("Estimated BPM")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bland_altman(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth = np.asarray([row["bpm0"] for row in rows], dtype=float)
    est = np.asarray([row["estimated_bpm"] for row in rows], dtype=float)
    mask = np.isfinite(truth) & np.isfinite(est)
    mean_values = (truth[mask] + est[mask]) / 2.0
    diff = est[mask] - truth[mask]
    ba = bland_altman(est, truth)
    plt.figure(figsize=(6, 4))
    plt.scatter(mean_values, diff, s=8, alpha=0.45)
    for y, label, color in [
        (ba["bias"], "bias", "black"),
        (ba["loa_lower"], "lower LoA", "tab:red"),
        (ba["loa_upper"], "upper LoA", "tab:red"),
    ]:
        if np.isfinite(y):
            plt.axhline(y, color=color, linewidth=1.0, linestyle="--" if label != "bias" else "-")
    plt.title(title)
    plt.xlabel("Mean of estimated and BPM0")
    plt.ylabel("Estimated - BPM0 (BPM)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def group_rows(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def summarize_window_rows(rows: list[dict[str, object]], group_label: str, group_value: str) -> dict[str, object]:
    truth = np.asarray([row["bpm0"] for row in rows], dtype=float)
    est = np.asarray([row["estimated_bpm"] for row in rows], dtype=float)
    metrics = compare_to_bpm0(est, truth)
    return {group_label: group_value, **metrics}


def weighted_nanmean(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float:
    values = np.asarray([float(row.get(value_key, math.nan)) for row in rows], dtype=float)
    weights = np.asarray([float(row.get(weight_key, 0.0)) for row in rows], dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        mask = np.isfinite(values)
        return float(np.mean(values[mask])) if np.any(mask) else math.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def summarize_beat_rows(rows: list[dict[str, object]], group_label: str, group_value: str) -> dict[str, object]:
    """Aggregate derived beat-level R-peak and IBI diagnostics."""
    tp = int(sum(int(row.get("rpeak_tp", 0)) for row in rows))
    fp = int(sum(int(row.get("rpeak_fp", 0)) for row in rows))
    fn = int(sum(int(row.get("rpeak_fn", 0)) for row in rows))
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    return {
        group_label: group_value,
        "n_records": len(rows),
        "n_reference_rpeaks": int(sum(int(row.get("n_reference_rpeaks", 0)) for row in rows)),
        "n_estimated_rpeaks": int(sum(int(row.get("n_estimated_rpeaks", 0)) for row in rows)),
        "rpeak_tp": tp,
        "rpeak_fp": fp,
        "rpeak_fn": fn,
        "rpeak_precision": float(precision),
        "rpeak_recall": float(recall),
        "rpeak_f1": float(f1),
        "rpeak_mean_abs_timing_error_ms": weighted_nanmean(rows, "rpeak_mean_abs_timing_error_ms", "rpeak_tp"),
        "n_ibi_matched": int(sum(int(row.get("n_ibi_matched", 0)) for row in rows)),
        "ibi_error_mae_ms": weighted_nanmean(rows, "ibi_error_mae_ms", "n_ibi_matched"),
        "ibi_error_rmse_ms": weighted_nanmean(rows, "ibi_error_rmse_ms", "n_ibi_matched"),
        "ibi_error_mean_ms": weighted_nanmean(rows, "ibi_error_mean_ms", "n_ibi_matched"),
        "ibi_error_sd_ms": weighted_nanmean(rows, "ibi_error_sd_ms", "n_ibi_matched"),
    }


def load_npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if value.shape == () else value


def save_peak_result(result: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rpeaks_samples=np.asarray(result["rpeaks_samples"], dtype=np.int64),
        rpeaks_seconds=np.asarray(result["rpeaks_seconds"], dtype=np.float32),
        ibi_ms=np.asarray(result["ibi_ms"], dtype=np.float32),
        instant_bpm=np.asarray(result["instant_bpm"], dtype=np.float32),
        ibi_outlier_mask=np.asarray(result["ibi_outlier_mask"], dtype=bool),
        candidate_peaks=np.asarray(result["candidate_peaks"], dtype=np.int64),
        candidate_scores=np.asarray(result["candidate_scores"], dtype=np.float32),
        prior_centers_s=np.asarray(result["prior_centers_s"], dtype=np.float32),
        prior_hr_bpm=np.asarray(result["prior_hr_bpm"], dtype=np.float32),
        prior_ibi_ms=np.asarray(result["prior_ibi_ms"], dtype=np.float32),
        n_pan_tompkins_candidates=int(result["n_pan_tompkins_candidates"]),
        n_wavelet_candidates=int(result["n_wavelet_candidates"]),
        n_merged_candidates=int(result["n_merged_candidates"]),
        beam_width=int(result["beam_width"]),
        sigma_hr_ms=float(result["sigma_hr_ms"]),
    )


def run_file(
    npz_path: Path,
    output_dir: Path,
    reference_cache: dict[str, dict[str, object]],
    beam_width: int,
    sigma_hr_ms: float,
    make_plots: bool,
    skip_beat_eval: bool,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        noisy_ecg = np.asarray(data["noisy_ecg"], dtype=np.float32)
        clean_ecg = np.asarray(data["clean_ecg"], dtype=np.float32)
        bpm0 = np.asarray(data["bpm0"], dtype=np.float32)
        window_start_s = np.asarray(data["window_start_s"], dtype=np.float32)
        window_end_s = np.asarray(data["window_end_s"], dtype=np.float32)
        fs = float(load_npz_scalar(data, "target_fs", FS))
        record_name = str(load_npz_scalar(data, "record_name", npz_path.stem))
        subject_id = str(load_npz_scalar(data, "subject_id", ""))
        treadmill_type = str(load_npz_scalar(data, "treadmill_type", ""))
        snr_db = float(load_npz_scalar(data, "target_snr_db", math.nan))
        achieved_snr_db = float(load_npz_scalar(data, "achieved_snr_db", math.nan))

    LOGGER.info("Running GRIP-IBI: %s at %g dB", record_name, snr_db)
    result = estimate_rpeaks_grip(noisy_ecg, fs=fs, beam_width=beam_width, sigma_hr_ms=sigma_hr_ms)
    estimated_bpm, rr_counts = compute_bpm_windows(result["rpeaks_samples"], window_start_s, window_end_s, fs)
    metrics = compare_to_bpm0(estimated_bpm, bpm0)

    peak_result_path = output_dir / "peak_results" / f"{record_name}_snr_{snr_db:g}db_peaks.npz".replace("-", "neg")
    save_peak_result(result, peak_result_path)

    record_row: dict[str, object] = {
        "record_name": record_name,
        "subject_id": subject_id,
        "treadmill_type": treadmill_type,
        "target_snr_db": snr_db,
        "achieved_snr_db": achieved_snr_db,
        "n_rpeaks": int(np.asarray(result["rpeaks_samples"]).size),
        "n_ibi": int(np.asarray(result["ibi_ms"]).size),
        "n_candidates": int(result["n_merged_candidates"]),
        "n_pan_tompkins_candidates": int(result["n_pan_tompkins_candidates"]),
        "n_wavelet_candidates": int(result["n_wavelet_candidates"]),
        "peak_result_path": str(peak_result_path),
        **metrics,
    }

    beat_row: dict[str, object] | None = None
    if not skip_beat_eval:
        if record_name not in reference_cache:
            LOGGER.info("Deriving clean-ECG reference peaks for %s", record_name)
            reference_cache[record_name] = estimate_rpeaks_grip(clean_ecg, fs=fs, beam_width=beam_width, sigma_hr_ms=sigma_hr_ms)
        ref_result = reference_cache[record_name]
        match = match_rpeaks(ref_result["rpeaks_samples"], result["rpeaks_samples"], fs=fs)
        ibi_metrics = ibi_error_from_matched_peaks(ref_result["rpeaks_samples"], result["rpeaks_samples"], match, fs=fs)
        beat_row = {
            "record_name": record_name,
            "subject_id": subject_id,
            "treadmill_type": treadmill_type,
            "target_snr_db": snr_db,
            "reference_source": "derived_from_clean_ecg_grip",
            "n_reference_rpeaks": int(np.asarray(ref_result["rpeaks_samples"]).size),
            "n_estimated_rpeaks": int(np.asarray(result["rpeaks_samples"]).size),
            "rpeak_tp": int(match["rpeak_tp"]),
            "rpeak_fp": int(match["rpeak_fp"]),
            "rpeak_fn": int(match["rpeak_fn"]),
            "rpeak_precision": float(match["rpeak_precision"]),
            "rpeak_recall": float(match["rpeak_recall"]),
            "rpeak_f1": float(match["rpeak_f1"]),
            "rpeak_mean_abs_timing_error_ms": float(match["rpeak_mean_abs_timing_error_ms"]),
            **ibi_metrics,
        }
        record_row.update({k: v for k, v in beat_row.items() if k not in {"record_name", "subject_id", "treadmill_type", "target_snr_db"}})

    window_rows: list[dict[str, object]] = []
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

    if make_plots:
        snr_label = f"snr_{snr_db:g}db".replace("-", "neg")
        plot_bpm_timeline(
            record_name,
            snr_db,
            window_start_s,
            bpm0,
            estimated_bpm,
            output_dir / "plots" / "by_record" / snr_label / f"{record_name}_bpm_timeline.png",
        )
        plot_example_peaks(
            record_name,
            snr_db,
            noisy_ecg,
            np.asarray(result["rpeaks_samples"], dtype=np.int64),
            fs,
            output_dir / "plots" / "by_record" / snr_label / f"{record_name}_example_peaks.png",
        )

    return record_row, window_rows, beat_row


def run_pipeline(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_files = sorted((processed_dir / "noisy_records").glob("snr_*db/*.npz"))
    if args.snrs is not None:
        requested = {float(snr) for snr in args.snrs}
        filtered = []
        for path in noisy_files:
            with np.load(path, allow_pickle=True) as data:
                snr_db = float(load_npz_scalar(data, "target_snr_db", math.nan))
            if snr_db in requested:
                filtered.append(path)
        noisy_files = filtered
    if args.max_files is not None:
        noisy_files = noisy_files[: args.max_files]
    if not noisy_files:
        raise FileNotFoundError(f"No noisy .npz files found under {processed_dir / 'noisy_records'}")

    LOGGER.info("Found %d noisy records", len(noisy_files))
    reference_cache: dict[str, dict[str, object]] = {}
    record_rows: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []
    beat_rows: list[dict[str, object]] = []

    for npz_path in noisy_files:
        record_row, window_rows, beat_row = run_file(
            npz_path=npz_path,
            output_dir=output_dir,
            reference_cache=reference_cache,
            beam_width=args.beam_width,
            sigma_hr_ms=args.sigma_hr_ms,
            make_plots=not args.no_plots,
            skip_beat_eval=args.skip_beat_eval,
        )
        record_rows.append(record_row)
        all_window_rows.extend(window_rows)
        if beat_row is not None:
            beat_rows.append(beat_row)

    by_snr = [
        summarize_window_rows(rows, "target_snr_db", snr)
        for snr, rows in sorted(group_rows(all_window_rows, "target_snr_db").items(), key=lambda item: float(item[0]))
    ]
    by_type = [
        summarize_window_rows(rows, "treadmill_type", treadmill_type)
        for treadmill_type, rows in sorted(group_rows(all_window_rows, "treadmill_type").items())
    ]
    by_snr_type = []
    for snr, snr_rows in sorted(group_rows(all_window_rows, "target_snr_db").items(), key=lambda item: float(item[0])):
        for treadmill_type, rows in sorted(group_rows(snr_rows, "treadmill_type").items()):
            by_snr_type.append({"target_snr_db": snr, **summarize_window_rows(rows, "treadmill_type", treadmill_type)})

    overall = summarize_window_rows(all_window_rows, "group", "overall")

    beat_overall = summarize_beat_rows(beat_rows, "group", "overall") if beat_rows else {}
    beat_by_snr = [
        summarize_beat_rows(rows, "target_snr_db", snr)
        for snr, rows in sorted(group_rows(beat_rows, "target_snr_db").items(), key=lambda item: float(item[0]))
    ] if beat_rows else []
    beat_by_type = [
        summarize_beat_rows(rows, "treadmill_type", treadmill_type)
        for treadmill_type, rows in sorted(group_rows(beat_rows, "treadmill_type").items())
    ] if beat_rows else []

    write_csv(record_rows, output_dir / "metrics_by_record.csv")
    write_csv(beat_rows, output_dir / "beat_level_metrics.csv")
    write_csv([beat_overall], output_dir / "beat_metrics_overall.csv")
    write_csv(beat_by_snr, output_dir / "beat_metrics_by_snr.csv")
    write_csv(beat_by_type, output_dir / "beat_metrics_by_treadmill_type.csv")
    write_csv(all_window_rows, output_dir / "window_predictions.csv")
    write_csv(by_snr, output_dir / "metrics_by_snr.csv")
    write_csv(by_type, output_dir / "metrics_by_treadmill_type.csv")
    write_csv(by_snr_type, output_dir / "metrics_by_snr_and_treadmill_type.csv")

    if not args.no_plots:
        for snr, rows in sorted(group_rows(all_window_rows, "target_snr_db").items(), key=lambda item: float(item[0])):
            snr_label = f"snr_{float(snr):g}db".replace("-", "neg")
            plot_scatter(rows, output_dir / "plots" / "summary" / f"{snr_label}_pearson_scatter.png", f"Pearson scatter ({float(snr):g} dB)")
            plot_bland_altman(rows, output_dir / "plots" / "summary" / f"{snr_label}_bland_altman.png", f"Bland-Altman ({float(snr):g} dB)")
        plot_scatter(all_window_rows, output_dir / "plots" / "summary" / "overall_pearson_scatter.png", "Pearson scatter (overall)")
        plot_bland_altman(all_window_rows, output_dir / "plots" / "summary" / "overall_bland_altman.png", "Bland-Altman (overall)")

    summary = {
        "config": {
            "processed_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "beam_width": int(args.beam_width),
            "sigma_hr_ms": float(args.sigma_hr_ms),
            "rr_min_ms": RR_MIN_MS,
            "rr_max_ms": RR_MAX_MS,
            "window_sec": WINDOW_SEC,
            "window_step_sec": WINDOW_STEP_SEC,
            "skip_beat_eval": bool(args.skip_beat_eval),
        },
        "overall": overall,
        "by_snr": by_snr,
        "by_treadmill_type": by_type,
        "by_snr_and_treadmill_type": by_snr_type,
        "beat_overall": beat_overall,
        "beat_by_snr": beat_by_snr,
        "beat_by_treadmill_type": beat_by_type,
        "n_records": len(record_rows),
        "n_windows": len(all_window_rows),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2)

    LOGGER.info("GRIP-IBI evaluation complete: %s", output_dir)
    LOGGER.info("Overall MAE %.3f BPM, RMSE %.3f BPM, Pearson r %.3f", overall["mae_bpm"], overall["rmse_bpm"], overall["pearson_r"])


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    hrv_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=hrv_dir / "processed_dataset")
    parser.add_argument("--output-dir", type=Path, default=hrv_dir / "grip_results")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--sigma-hr-ms", type=float, default=200.0)
    parser.add_argument("--snrs", type=float, nargs="*", default=None, help="Optional subset of SNR levels to process.")
    parser.add_argument("--max-files", type=int, default=None, help="Debug option to process only the first N noisy files.")
    parser.add_argument("--skip-beat-eval", action="store_true", help="Skip derived clean-reference R-peak and IBI diagnostics.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    run_pipeline(args)


if __name__ == "__main__":
    main()
