"""
CkCk Hoax Detection AI — Source Package
========================================
Privacy-aware Indonesian hoax detection system with PII filtering.

Modules:
    - model: Model definition, 4-status output, and inference
    - dataset: Dataset class and data loading utilities
    - pii_filter: PII detection and redaction pipeline
    - preprocessing: Indonesian text cleaning and normalization
    - trainer: Training loop and evaluation
    - rule_detector: Manipulative pattern detection (explainability)
    - ocr_engine: OCR text extraction from images
    - utils: Shared utility functions
"""

from .pii_filter import PIIFilter
from .preprocessing import TextPreprocessor
from .model import HoaxDetector
from .dataset import HoaxDataset
from .rule_detector import RuleBasedDetector
from .ocr_engine import OCREngine

__version__ = "0.2.0"
__all__ = [
    "PIIFilter",
    "TextPreprocessor",
    "HoaxDetector",
    "HoaxDataset",
    "RuleBasedDetector",
    "OCREngine",
]
