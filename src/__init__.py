"""
CkCk Hoax Detection AI — Source Package
========================================
Privacy-aware Indonesian hoax detection system with PII filtering.

Modules:
    - model: Model definition, loading, and inference
    - dataset: Dataset class and data loading utilities
    - pii_filter: PII detection and redaction pipeline
    - preprocessing: Indonesian text cleaning and normalization
    - rule_based: Rule-based manipulative pattern detector
    - output_engine: Final scoring and human-readable output (Bahasa Indonesia)
    - inferencer: Full inference pipeline orchestrator
    - trainer: Training loop and evaluation
    - utils: Shared utility functions
"""

from .pii_filter import PIIFilter
from .preprocessing import TextPreprocessor
from .model import HoaxDetector
from .dataset import HoaxDataset
from .rule_based import compute_support_score, detect_patterns
from .output_engine import generate_output
from .inferencer import pii_filter, inference_pipeline, CkCkInferencer, run_ckck_inference

__version__ = "0.1.0"
__all__ = [
    "PIIFilter",
    "TextPreprocessor",
    "HoaxDetector",
    "HoaxDataset",
    "compute_support_score",
    "detect_patterns",
    "generate_output",
    "pii_filter",
    "inference_pipeline",
    "CkCkInferencer",
    "run_ckck_inference",
]
