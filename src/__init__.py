"""
CkCk Hoax Detection AI — Source Package
========================================
Privacy-aware Indonesian hoax detection system with PII filtering.

Modules:
    - model: Model definition, loading, and inference
    - dataset: Dataset class and data loading utilities
    - pii_filter: PII detection and redaction pipeline
    - preprocessing: Indonesian text cleaning and normalization
    - trainer: Training loop and evaluation
    - utils: Shared utility functions
"""

from .pii_filter import PIIFilter
from .preprocessing import TextPreprocessor
from .model import HoaxDetector
from .dataset import HoaxDataset

__version__ = "0.1.0"
__all__ = ["PIIFilter", "TextPreprocessor", "HoaxDetector", "HoaxDataset"]
