"""
Output Engine — Final Scoring & Human-Readable Output
========================================================
Combines IndoBERT classifier confidence with rule-based support signals
to produce a final status and user-friendly explanation in Bahasa Indonesia.

Four output statuses:
  ✅ TERVERIFIKASI              — High confidence that the content is valid
  ⚠️  KONTEKS BERBEDA            — Moderate confidence; content may be valid
                                   but context might be misleading
  🔍 BELUM TERVERIFIKASI — WASPADAI — Classified as hoax with manipulative patterns
  ❓ BELUM TERVERIFIKASI — NETRAL  — Low confidence, no clear manipulative signal

Scoring formula:
  final_score = (0.7 × primary_score) + (0.3 × support_score)

All processing is offline — no API calls, no internet access.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rule_based import (
    RuleBasedResult,
    format_patterns_indonesian,
)


# ── Constants ─────────────────────────────────────────────────────────────

# Weight allocation for the composite score
PRIMARY_WEIGHT = 0.7
SUPPORT_WEIGHT = 0.3

# Threshold boundaries (applied on classifier confidence, NOT final_score)
THRESHOLD_VERIFIED     = 0.75   # confidence_valid ≥ 75% → TERVERIFIKASI
THRESHOLD_CONTEXT_DIFF = 0.50   # confidence_valid 50–75% → KONTEKS BERBEDA

# Status constants
STATUS_VERIFIED           = "TERVERIFIKASI"
STATUS_CONTEXT_DIFFERENT  = "KONTEKS BERBEDA"
STATUS_UNVERIFIED_ALERT   = "BELUM TERVERIFIKASI — WASPADAI"
STATUS_UNVERIFIED_NEUTRAL = "BELUM TERVERIFIKASI — NETRAL"


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class OutputResult:
    """Structured output from the output engine."""
    status: str                      # One of the four statuses
    final_score: float               # Composite score (0.0–1.0)
    confidence_valid: float          # Classifier confidence for "VALID"
    confidence_hoax: float           # Classifier confidence for "HOAX"
    support_score: float             # Rule-based manipulation score
    penjelasan: str                  # Human-readable explanation (Bahasa Indonesia)
    pola_terdeteksi: list[str]       # Detected manipulative pattern categories
    pii_disensor: bool               # Whether PII was redacted


# ── Core Functions ────────────────────────────────────────────────────────

def compute_final_score(primary_score: float, support_score: float) -> float:
    """
    Combine IndoBERT classifier score with rule-based support score.

    Formula: final_score = (0.7 × primary_score) + (0.3 × support_score)

    Args:
        primary_score: Hoax confidence from the classifier (0.0–1.0).
                       Higher means more likely hoax.
        support_score: Rule-based manipulation score (0.0–1.0).
                       Higher means more manipulative patterns found.

    Returns:
        Composite float score in [0.0, 1.0].
    """
    score = (PRIMARY_WEIGHT * primary_score) + (SUPPORT_WEIGHT * support_score)
    return round(min(max(score, 0.0), 1.0), 4)


def determine_status(
    confidence_valid: float,
    confidence_hoax: float,
    support_score: float,
) -> str:
    """
    Determine one of the four output statuses based on classifier confidence
    and rule-based support score.

    Decision logic:
      1. confidence_valid ≥ 0.75  → TERVERIFIKASI
      2. confidence_valid 0.50–0.75 → KONTEKS BERBEDA
      3. confidence_hoax ≥ 0.50 AND support_score > 0 → BELUM TERVERIFIKASI — WASPADAI
      4. Otherwise → BELUM TERVERIFIKASI — NETRAL

    Args:
        confidence_valid: Classifier probability for "VALID".
        confidence_hoax: Classifier probability for "HOAX".
        support_score: Rule-based manipulation score.

    Returns:
        One of the four status strings.
    """
    if confidence_valid >= THRESHOLD_VERIFIED:
        return STATUS_VERIFIED

    if confidence_valid >= THRESHOLD_CONTEXT_DIFF:
        return STATUS_CONTEXT_DIFFERENT

    if confidence_hoax >= THRESHOLD_CONTEXT_DIFF and support_score > 0.0:
        return STATUS_UNVERIFIED_ALERT

    return STATUS_UNVERIFIED_NEUTRAL


def generate_explanation(
    status: str,
    confidence_valid: float,
    confidence_hoax: float,
    rule_result: RuleBasedResult | None,
    pii_disensor: bool,
) -> str:
    """
    Generate a human-readable explanation in Bahasa Indonesia.

    The explanation is written for non-technical users and avoids jargon.
    It describes WHY the system reached its conclusion.

    Args:
        status: The determined status string.
        confidence_valid: Classifier confidence for "VALID".
        confidence_hoax: Classifier confidence for "HOAX".
        rule_result: RuleBasedResult from the pattern detector (may be None).
        pii_disensor: Whether PII redaction was applied.

    Returns:
        Indonesian-language explanation string.
    """
    parts: list[str] = []

    # ── Status-specific opening ──────────────────────────────────────
    if status == STATUS_VERIFIED:
        parts.append(
            "Konten ini memiliki kesesuaian tinggi dengan informasi yang terverifikasi. "
            f"Tingkat keyakinan: {confidence_valid * 100:.0f}%."
        )

    elif status == STATUS_CONTEXT_DIFFERENT:
        parts.append(
            "Konten ini mungkin benar secara substansi, namun konteks penyajiannya "
            "perlu diperhatikan. Kami menyarankan untuk memeriksa sumber aslinya. "
            f"Tingkat keyakinan valid: {confidence_valid * 100:.0f}%."
        )

    elif status == STATUS_UNVERIFIED_ALERT:
        parts.append(
            "Konten ini memiliki karakteristik kuat sebagai konten manipulatif. "
            f"Tingkat kecurigaan: {confidence_hoax * 100:.0f}%."
        )

    else:  # STATUS_UNVERIFIED_NEUTRAL
        parts.append(
            "Konten ini belum dapat diverifikasi. Sistem tidak menemukan "
            "bukti kuat bahwa konten ini benar maupun palsu. "
            "Disarankan untuk tetap berhati-hati dan mencari sumber informasi tambahan."
        )

    # ── Manipulative pattern details ─────────────────────────────────
    if rule_result and rule_result.patterns_found:
        pattern_desc = format_patterns_indonesian(rule_result)
        parts.append(f"Ditemukan {pattern_desc}")

    # ── PII notice ───────────────────────────────────────────────────
    if pii_disensor:
        parts.append(
            "Catatan: Data pribadi (PII) yang terdeteksi dalam teks telah "
            "disensor untuk perlindungan privasi."
        )

    return " ".join(parts)


def generate_output(
    primary_score: float,
    support_score: float,
    context: dict,
) -> dict:
    """
    Combine all scores and context into the final output dictionary.

    This is the main entry point for the output engine. It computes the
    final composite score, determines the status, and generates a
    human-readable explanation in Bahasa Indonesia.

    Args:
        primary_score: Hoax confidence from the classifier (0.0–1.0).
                       This is the `confidence_hoax` value.
        support_score: Rule-based manipulation score (0.0–1.0).
        context: Dictionary with additional pipeline data. Expected keys:
            - "confidence_valid" (float): Classifier confidence for VALID.
            - "confidence_hoax" (float): Classifier confidence for HOAX.
            - "rule_result" (RuleBasedResult | None): Pattern detection output.
            - "pii_disensor" (bool): Whether PII was redacted.
            - "classifier_label" (str): Raw label from classifier ("HOAX"/"VALID").

    Returns:
        dict with keys:
            - status (str)
            - final_score (float)
            - confidence_valid (float)
            - confidence_hoax (float)
            - support_score (float)
            - penjelasan (str)
            - pola_terdeteksi (list[str])
            - pii_disensor (bool)
    """
    # Extract context values
    confidence_valid = context.get("confidence_valid", 1.0 - primary_score)
    confidence_hoax  = context.get("confidence_hoax", primary_score)
    rule_result      = context.get("rule_result", None)
    pii_disensor     = context.get("pii_disensor", False)

    # Compute composite score
    final_score = compute_final_score(primary_score, support_score)

    # Determine status
    status = determine_status(confidence_valid, confidence_hoax, support_score)

    # Generate explanation
    penjelasan = generate_explanation(
        status=status,
        confidence_valid=confidence_valid,
        confidence_hoax=confidence_hoax,
        rule_result=rule_result,
        pii_disensor=pii_disensor,
    )

    # Collect detected pattern categories
    pola_terdeteksi = rule_result.categories_found if rule_result else []

    return {
        "status":           status,
        "final_score":      final_score,
        "confidence_valid":  round(confidence_valid, 4),
        "confidence_hoax":   round(confidence_hoax, 4),
        "support_score":    round(support_score, 4),
        "penjelasan":       penjelasan,
        "pola_terdeteksi":  pola_terdeteksi,
        "pii_disensor":     pii_disensor,
    }


# ── Module Self-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.rule_based import RuleBasedResult, PatternMatch

    # Simulate different scenarios
    scenarios = [
        {
            "name": "Verified valid news",
            "primary_score": 0.10,
            "support_score": 0.0,
            "context": {
                "confidence_valid": 0.90,
                "confidence_hoax": 0.10,
                "rule_result": RuleBasedResult(0.0, [], [], {"urgency": 0, "fear": 0, "attribution": 0}),
                "pii_disensor": False,
            },
        },
        {
            "name": "Context different",
            "primary_score": 0.40,
            "support_score": 0.15,
            "context": {
                "confidence_valid": 0.60,
                "confidence_hoax": 0.40,
                "rule_result": RuleBasedResult(0.15, [
                    PatternMatch("urgency", "VIRAL", 0.10, []),
                ], ["urgency"], {"urgency": 1, "fear": 0, "attribution": 0}),
                "pii_disensor": False,
            },
        },
        {
            "name": "Hoax with manipulative patterns",
            "primary_score": 0.88,
            "support_score": 0.45,
            "context": {
                "confidence_valid": 0.12,
                "confidence_hoax": 0.88,
                "rule_result": RuleBasedResult(0.45, [
                    PatternMatch("urgency", "SEBARKAN", 0.15, []),
                    PatternMatch("fear", "BAHAYA", 0.12, []),
                    PatternMatch("attribution", "kata pemerintah", 0.15, []),
                ], ["attribution", "fear", "urgency"], {"urgency": 1, "fear": 1, "attribution": 1}),
                "pii_disensor": True,
            },
        },
        {
            "name": "Neutral — low confidence, no patterns",
            "primary_score": 0.55,
            "support_score": 0.0,
            "context": {
                "confidence_valid": 0.45,
                "confidence_hoax": 0.55,
                "rule_result": RuleBasedResult(0.0, [], [], {"urgency": 0, "fear": 0, "attribution": 0}),
                "pii_disensor": False,
            },
        },
    ]

    for s in scenarios:
        print(f"═══ {s['name']} ═══")
        output = generate_output(s["primary_score"], s["support_score"], s["context"])
        for k, v in output.items():
            print(f"  {k}: {v}")
        print()
