"""
CkCk Hoax Detection AI — Source Package
========================================
Privacy-aware Indonesian hoax detection system with PII filtering.

Modules:
    - inferencer      : Core inference engine (ONNX, OCR, confidence scoring)
    - mock_inferencer : Mock inference untuk testing paralel (interface identik)
    - model           : Model wrapper PyTorch (untuk training)
    - dataset         : Dataset class and data loading utilities
    - pii_filter      : PII detection and redaction pipeline
    - preprocessing   : Indonesian text cleaning and normalization
    - manipulative_detector : Rule-based pola linguistik manipulatif
    - trainer         : Training loop and evaluation
    - utils           : Shared utility functions
"""

from .pii_filter import PIIFilter
from .preprocessing import TextPreprocessor
from .model import HoaxDetector
from .dataset import HoaxDataset
from .inferencer import (
    load_models,
    prepare_input,
    run_classifier,
    compute_confidence,
    run_ckck_inference,
)

__version__ = "0.2.0"
__all__ = [
    "PIIFilter",
    "TextPreprocessor",
    "HoaxDetector",
    "HoaxDataset",
    "load_models",
    "prepare_input",
    "run_classifier",
    "compute_confidence",
    "run_ckck_inference",
]
