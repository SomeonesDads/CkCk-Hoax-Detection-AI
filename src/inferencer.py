"""
Inferencer — Full Inference Pipeline
=======================================
Wraps all components into a single callable pipeline:

  Input Text
    → PII Filter         (redact personal data)
    → Preprocessor       (clean text for model)
    → IndoBERT Classifier (primary decision maker)
    → Rule-Based Detector (enrich explanation)
    → Output Engine       (composite score + 4 statuses + Bahasa Indonesia output)

All processing is offline — no API calls, no internet access.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from src.pii_filter import PIIFilter
from src.preprocessing import TextPreprocessor
from src.model import HoaxDetector
from src.rule_based import detect_patterns, compute_support_score, get_pattern_labels
from src.output_engine import generate_output
from src.utils import load_config


# ── Standalone PII Filter Function ────────────────────────────────────────

def pii_filter(text: str) -> str:
    """
    Detect and redact personally identifiable information (PII) from text.

    Replaces the following PII types with descriptive redaction tags:
      - NIK (16-digit Indonesian national ID)  → [REDACTED_NIK]
      - Phone numbers (+62xx / 08xx)            → [REDACTED_PHONE]
      - Email addresses                         → [REDACTED_EMAIL]
      - Bank account numbers (10–16 digits)     → [REDACTED_ACCOUNT]

    This is a convenience wrapper around PIIFilter for use as a standalone
    function. For richer output (positions, details), use PIIFilter directly.

    Processing order: NIK → Email → Phone → Bank Account (most specific first).
    Each pattern is applied sequentially. Once a region is replaced by a tag
    (e.g., [REDACTED_NIK]), the tag text will not match subsequent numeric
    patterns, preventing double-redaction naturally.

    Args:
        text: Raw input text.

    Returns:
        Text with all detected PII replaced by redaction tags.
    """
    import re

    # ── NIK ──────────────────────────────────────────────────────────
    # Indonesian NIK: 16 digits total.
    # Format: PPKKCC-DDMMYY-SSSS
    #   PP  = province code (2 digits)
    #   KK  = city/regency code (2 digits)
    #   CC  = sub-district code (2 digits)
    #   DD  = day of birth (01-31 for male, 41-71 for female = day+40)
    #   MM  = month (01-12)
    #   YY  = year (2 digits)
    #   SSSS = serial (4 digits)
    nik_pattern = re.compile(
        r'\b\d{6}'                                # 6-digit area code
        r'(?:0[1-9]|[12]\d|3[01]|4[1-9]|[56]\d|7[01])'  # day: 01-31 or 41-71
        r'(?:0[1-9]|1[0-2])'                     # month: 01-12
        r'\d{2}'                                  # year: 2 digits
        r'\d{4}'                                  # serial: 4 digits
        r'\b'
    )
    text = nik_pattern.sub("[REDACTED_NIK]", text)

    # ── Email ────────────────────────────────────────────────────────
    email_pattern = re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    )
    text = email_pattern.sub("[REDACTED_EMAIL]", text)

    # ── Phone ────────────────────────────────────────────────────────
    # Indonesian mobile: +62/62/0 followed by 8xx-xxxx-xxxx
    # Indonesian landline: +62/62/0 followed by area code + number
    phone_pattern = re.compile(
        r'(?:\+62|62|0)[\s\-]?'
        r'(?:8\d[\s\-]?\d{3,4}[\s\-]?\d{3,4}'
        r'|(?:21|22|24|31|61|71|411|421|451|471|541|551|561|711)\s?\d{5,8})'
    )
    text = phone_pattern.sub("[REDACTED_PHONE]", text)

    # ── Bank Account ─────────────────────────────────────────────────
    # 10–16 digit standalone number. Applied last so NIK (already replaced
    # by [REDACTED_NIK]) won't be matched again.
    bank_pattern = re.compile(r'\b\d{10,16}\b')
    text = bank_pattern.sub("[REDACTED_ACCOUNT]", text)

    return text


# ── Full Pipeline ─────────────────────────────────────────────────────────

def inference_pipeline(input_text: str) -> dict:
    """
    Execute the full CkCk hoax detection inference pipeline.

    Pipeline steps:
      1. PII Filter      — Redact personal data before any model processing
      2. Preprocessing    — Clean text (remove URLs, HTML, normalize, etc.)
      3. IndoBERT Classify — Primary hoax/valid decision
      4. Rule-Based Detect — Scan for manipulative language patterns
      5. Output Engine     — Combine scores, determine status, generate explanation

    This function initializes all components internally. For repeated calls,
    use the CkCkInferencer class instead (component reuse = faster).

    Args:
        input_text: Raw input text (caption, OCR result, etc.)

    Returns:
        dict with keys:
            - status (str): One of 4 statuses
            - final_score (float): Composite score
            - confidence_valid (float): Classifier confidence for VALID
            - confidence_hoax (float): Classifier confidence for HOAX
            - support_score (float): Rule-based manipulation score
            - penjelasan (str): Human-readable explanation (Bahasa Indonesia)
            - pola_terdeteksi (list[str]): Detected pattern categories
            - pii_disensor (bool): Whether PII was redacted
            - inference_time_ms (float): Total inference time
            - original_text (str): Original input
            - pii_filtered_text (str): Text after PII redaction
    """
    inferencer = CkCkInferencer.from_config("config.yaml")
    return inferencer.run(input_text)


class CkCkInferencer:
    """
    Reusable inference pipeline that keeps all components loaded in memory.

    Usage:
        inferencer = CkCkInferencer.from_config("config.yaml")
        result = inferencer.run("Berita ini sangat mencurigakan!!")
    """

    def __init__(
        self,
        detector: HoaxDetector,
        pii: PIIFilter,
        preprocessor: TextPreprocessor,
    ):
        self.detector = detector
        self.pii = pii
        self.preprocessor = preprocessor

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "CkCkInferencer":
        """
        Build the inferencer from the project config file.

        Loads IndoBERT model (fine-tuned checkpoint if available, otherwise
        pre-trained), PII filter, and text preprocessor.
        """
        config = load_config(config_path)

        # ── Model ────────────────────────────────────────────────────
        detector = HoaxDetector(
            model_name=config["model"]["name"],
            num_labels=config["model"]["num_labels"],
            max_length=config["model"]["max_length"],
            device=config.get("inference", {}).get("device", "cpu"),
        )
        model_path = os.path.join(config["paths"]["model_dir"], "best_model")
        if os.path.exists(model_path):
            detector.load_finetuned(model_path)
        else:
            detector.load_pretrained()

        # ── PII Filter ───────────────────────────────────────────────
        pii = PIIFilter(
            mask_char=config["pii_filter"]["mask_char"],
            enabled_types=config["pii_filter"]["types"],
        )

        # ── Preprocessor ─────────────────────────────────────────────
        preprocessor = TextPreprocessor(use_stemmer=False)

        return cls(detector=detector, pii=pii, preprocessor=preprocessor)

    def run(self, input_text: str, verbose: bool = False) -> dict:
        """
        Execute the full pipeline on a single input text.

        Args:
            input_text: Raw text to analyze.
            verbose: If True, print step-by-step progress.

        Returns:
            Complete result dict (see inference_pipeline docstring).
        """
        start_time = time.time()

        # ── Step 1: PII Filter ───────────────────────────────────────
        pii_result = self.pii.filter(input_text)
        safe_text = pii_result["filtered_text"]
        pii_disensor = pii_result["pii_found"]

        if verbose:
            print(f"[1/5] PII Filter: {pii_result['pii_count']} items redacted")

        # ── Step 2: Preprocessing ────────────────────────────────────
        cleaned_text = self.preprocessor.clean(safe_text)

        if verbose:
            print(f"[2/5] Preprocessor: text cleaned ({len(cleaned_text)} chars)")

        # ── Step 3: IndoBERT Classification ──────────────────────────
        prediction = self.detector.predict(cleaned_text)
        confidence_valid = prediction["probabilities"]["VALID"]
        confidence_hoax  = prediction["probabilities"]["HOAX"]

        # primary_score = hoax confidence (higher = more likely hoax)
        primary_score = confidence_hoax

        if verbose:
            print(f"[3/5] Classifier: {prediction['label']} "
                  f"(valid={confidence_valid:.2%}, hoax={confidence_hoax:.2%})")

        # ── Step 4: Rule-Based Pattern Detection ─────────────────────
        # Run on the PII-filtered text (NOT cleaned) to preserve casing
        # and punctuation patterns that are signal for manipulation.
        rule_result = detect_patterns(safe_text)
        support_score = rule_result.support_score

        if verbose:
            print(f"[4/5] Rule-based: score={support_score:.4f}, "
                  f"patterns={get_pattern_labels(rule_result)}")

        # ── Step 5: Output Engine ────────────────────────────────────
        context = {
            "confidence_valid": confidence_valid,
            "confidence_hoax":  confidence_hoax,
            "rule_result":      rule_result,
            "pii_disensor":     pii_disensor,
            "classifier_label": prediction["label"],
        }

        output = generate_output(
            primary_score=primary_score,
            support_score=support_score,
            context=context,
        )

        elapsed = time.time() - start_time

        if verbose:
            print(f"[5/5] Output: {output['status']} "
                  f"(final_score={output['final_score']:.4f})")

        # ── Compose final result ─────────────────────────────────────
        output["inference_time_ms"] = round(elapsed * 1000, 2)
        output["original_text"]     = input_text
        output["pii_filtered_text"] = safe_text
        output["cleaned_text"]      = cleaned_text
        output["classifier_label"]  = prediction["label"]
        output["classifier_confidence"] = prediction["confidence"]
        output["pii_details"]       = pii_result["details"]

        return output


# ── Convenience Function (matches README API) ────────────────────────────

def run_ckck_inference(
    raw_input: str,
    input_type: str = "caption",
    config_path: str = "config.yaml",
    verbose: bool = False,
) -> dict:
    """
    Public API matching the README example.

    Args:
        raw_input: Input text (caption) or image path (frame).
        input_type: "caption" for text, "frame" for image (OCR placeholder).
        config_path: Path to config.yaml.
        verbose: Print progress steps.

    Returns:
        Complete inference result dict.
    """
    if input_type == "frame":
        # OCR would be handled here by the OCR module (separate teammate).
        # For now, raise a clear error rather than silently failing.
        raise NotImplementedError(
            "OCR module (frame input) is handled by a separate module. "
            "Please use input_type='caption' with text input."
        )

    inferencer = CkCkInferencer.from_config(config_path)
    return inferencer.run(raw_input, verbose=verbose)


# ── Module Self-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test the standalone pii_filter function
    test_texts = [
        "NIK saya 3201234506780001 tolong dijaga.",
        "Hubungi 081234567890 atau email budi@gmail.com",
        "Transfer ke rekening 1234567890123456.",
        "Berita ini tidak mengandung data pribadi.",
    ]

    print("═══ Standalone pii_filter() Test ═══\n")
    for text in test_texts:
        filtered = pii_filter(text)
        print(f"  Input:  {text}")
        print(f"  Output: {filtered}")
        print()

    # Full pipeline test requires model — skip in self-test
    print("═══ Full pipeline requires model loading. ═══")
    print("Use inference.ipynb or call:")
    print("  from src.inferencer import run_ckck_inference")
    print("  result = run_ckck_inference('your text here')")
