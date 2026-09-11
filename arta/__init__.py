"""ARTA (Artifact-Robust Temporal Alignment) for ECG R-peak/IBI estimation."""

from .arta import build_arta_candidate_pool, estimate_rpeaks_arta

__version__ = "0.1.0"

__all__ = ["__version__", "build_arta_candidate_pool", "estimate_rpeaks_arta"]
