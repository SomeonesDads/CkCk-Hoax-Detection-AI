"""
Rule-Based Manipulative Pattern Detector
==========================================
Detects manipulative language patterns commonly found in Indonesian hoax content.

This module runs AFTER the IndoBERT classifier. It does NOT change the
classification decision — it provides supplementary evidence and enriches
the human-readable explanation.

Three categories of manipulative patterns:
  1. Urgency  — false urgency to pressure sharing (e.g., "SEGERA", "VIRAL")
  2. Fear     — fear-mongering language (e.g., "BAHAYA", "ANCAMAN")
  3. Attribution — unverified source claims (e.g., "kata pemerintah", "menurut ahli")

All processing is offline — no API calls, no internet access.
"""

import re
from dataclasses import dataclass, field


# ── Pattern Definitions ───────────────────────────────────────────────────

# Each pattern group: (compiled regex, display label, weight)
# Weights determine how strongly each match contributes to the support score.

URGENCY_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r'\bSEGERA\b',           re.IGNORECASE), "SEGERA",           0.12),
    (re.compile(r'\bVIRAL\b',            re.IGNORECASE), "VIRAL",            0.10),
    (re.compile(r'\bSEBARKAN\b',         re.IGNORECASE), "SEBARKAN",         0.15),
    (re.compile(r'\bBAGIKAN\b',          re.IGNORECASE), "BAGIKAN",          0.12),
    (re.compile(r'\bBREAKING\b',         re.IGNORECASE), "BREAKING",         0.10),
    (re.compile(r'\bURGENT\b',           re.IGNORECASE), "URGENT",           0.10),
    (re.compile(r'\bDARURAT\b',          re.IGNORECASE), "DARURAT",          0.10),
    (re.compile(r'sebelum\s+dihapus',    re.IGNORECASE), "sebelum dihapus",  0.15),
    (re.compile(r'jangan\s+sampai\s+terhapus', re.IGNORECASE), "jangan sampai terhapus", 0.15),
    (re.compile(r'share\s+sebelum',      re.IGNORECASE), "share sebelum",    0.12),
    (re.compile(r'harus\s+tau\b',        re.IGNORECASE), "harus tau",        0.08),
    (re.compile(r'wajib\s+baca\b',       re.IGNORECASE), "wajib baca",       0.08),
]

FEAR_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r'\bBAHAYA\b',           re.IGNORECASE), "BAHAYA",           0.12),
    (re.compile(r'\bANCAMAN\b',          re.IGNORECASE), "ANCAMAN",          0.10),
    (re.compile(r'\bKORBAN\b',           re.IGNORECASE), "KORBAN",           0.08),
    (re.compile(r'\bHANCUR\b',           re.IGNORECASE), "HANCUR",           0.10),
    (re.compile(r'\bMATI\b',             re.IGNORECASE), "MATI",             0.08),
    (re.compile(r'\bMENGERIKAN\b',       re.IGNORECASE), "MENGERIKAN",       0.10),
    (re.compile(r'\bMENAKUTKAN\b',       re.IGNORECASE), "MENAKUTKAN",       0.10),
    (re.compile(r'\bBENCAN[Aa]\b',       re.IGNORECASE), "BENCANA",          0.08),
    (re.compile(r'\bRACUN\b',            re.IGNORECASE), "RACUN",            0.10),
    (re.compile(r'\bTERCEMAR\b',         re.IGNORECASE), "TERCEMAR",         0.08),
    (re.compile(r'\bAWAS\b',             re.IGNORECASE), "AWAS",             0.08),
    (re.compile(r'\bWASPADA\b',          re.IGNORECASE), "WASPADA",          0.06),
]

ATTRIBUTION_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r'kata\s+pemerintah',          re.IGNORECASE), "kata pemerintah",          0.15),
    (re.compile(r'menurut\s+ahli',             re.IGNORECASE), "menurut ahli",             0.12),
    (re.compile(r'menurut\s+pakar',            re.IGNORECASE), "menurut pakar",             0.12),
    (re.compile(r'menurut\s+dokter',           re.IGNORECASE), "menurut dokter",            0.10),
    (re.compile(r'menurut\s+peneliti',         re.IGNORECASE), "menurut peneliti",          0.10),
    (re.compile(r'menurut\s+sumber\s+terpercaya', re.IGNORECASE), "menurut sumber terpercaya", 0.15),
    (re.compile(r'sudah\s+terbukti',           re.IGNORECASE), "sudah terbukti",            0.12),
    (re.compile(r'telah\s+terbukti',           re.IGNORECASE), "telah terbukti",            0.12),
    (re.compile(r'terbukti\s+secara\s+ilmiah', re.IGNORECASE), "terbukti secara ilmiah",    0.15),
    (re.compile(r'penelitian\s+membuktikan',   re.IGNORECASE), "penelitian membuktikan",    0.12),
    (re.compile(r'fakta\s+yang\s+disembunyikan', re.IGNORECASE), "fakta yang disembunyikan", 0.15),
]


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """A single matched manipulative pattern."""
    category: str          # "urgency", "fear", or "attribution"
    label: str             # Human-readable pattern name
    weight: float          # Contribution to the score
    positions: list[tuple[int, int]] = field(default_factory=list)  # (start, end)


@dataclass
class RuleBasedResult:
    """Aggregated result from the rule-based detector."""
    support_score: float                    # 0.0–1.0
    patterns_found: list[PatternMatch]      # All matched patterns
    categories_found: list[str]             # Unique categories triggered
    summary: dict[str, int]                 # Count per category


# ── Core Functions ────────────────────────────────────────────────────────

def _scan_patterns(
    text: str,
    patterns: list[tuple[re.Pattern, str, float]],
    category: str,
) -> list[PatternMatch]:
    """
    Scan text against a list of regex patterns and return matches.

    Args:
        text: The input text to scan.
        patterns: List of (compiled_regex, display_label, weight).
        category: Category name for these patterns.

    Returns:
        List of PatternMatch objects for every pattern found.
    """
    matches = []
    for regex, label, weight in patterns:
        found = list(regex.finditer(text))
        if found:
            positions = [(m.start(), m.end()) for m in found]
            matches.append(PatternMatch(
                category=category,
                label=label,
                weight=weight,
                positions=positions,
            ))
    return matches


def compute_support_score(text: str) -> float:
    """
    Compute a rule-based manipulation support score (0.0–1.0).

    Scans the input text for three categories of manipulative patterns:
      - Urgency: pressure language ("SEGERA", "VIRAL", "SEBARKAN", etc.)
      - Fear: fear-mongering language ("BAHAYA", "ANCAMAN", "KORBAN", etc.)
      - Attribution: unverified source claims ("kata pemerintah", "menurut ahli")

    The score is the sum of individual pattern weights, clamped to [0.0, 1.0].
    Higher values indicate stronger manipulative characteristics.

    This function is deterministic and fully offline.

    Args:
        text: Input text (ideally PII-filtered, pre-classifier text).

    Returns:
        Float score in range [0.0, 1.0].
    """
    result = detect_patterns(text)
    return result.support_score


def detect_patterns(text: str) -> RuleBasedResult:
    """
    Full pattern detection — returns score plus detailed pattern matches.

    Use this when you need both the score AND the list of detected patterns
    for generating human-readable explanations.

    Args:
        text: Input text to analyze.

    Returns:
        RuleBasedResult with score, matched patterns, categories, and summary.
    """
    if not text or not isinstance(text, str):
        return RuleBasedResult(
            support_score=0.0,
            patterns_found=[],
            categories_found=[],
            summary={"urgency": 0, "fear": 0, "attribution": 0},
        )

    all_matches: list[PatternMatch] = []

    # Scan each category
    all_matches.extend(_scan_patterns(text, URGENCY_PATTERNS,     "urgency"))
    all_matches.extend(_scan_patterns(text, FEAR_PATTERNS,        "fear"))
    all_matches.extend(_scan_patterns(text, ATTRIBUTION_PATTERNS, "attribution"))

    # Compute raw score as sum of weights, clamped to [0.0, 1.0]
    raw_score = sum(m.weight for m in all_matches)
    clamped_score = min(max(raw_score, 0.0), 1.0)

    # Category summary
    categories_found = sorted(set(m.category for m in all_matches))
    summary = {
        "urgency":     sum(1 for m in all_matches if m.category == "urgency"),
        "fear":        sum(1 for m in all_matches if m.category == "fear"),
        "attribution": sum(1 for m in all_matches if m.category == "attribution"),
    }

    return RuleBasedResult(
        support_score=round(clamped_score, 4),
        patterns_found=all_matches,
        categories_found=categories_found,
        summary=summary,
    )


def get_pattern_labels(result: RuleBasedResult) -> list[str]:
    """
    Extract human-readable labels from a RuleBasedResult.

    Args:
        result: Output from detect_patterns().

    Returns:
        List of pattern label strings, e.g. ["SEGERA", "BAHAYA", "kata pemerintah"].
    """
    return [m.label for m in result.patterns_found]


# ── Indonesian Display Helpers ────────────────────────────────────────────

CATEGORY_DISPLAY_ID = {
    "urgency":     "urgensi palsu",
    "fear":        "menakut-nakuti (fear-mongering)",
    "attribution": "atribusi tanpa sumber valid",
}


def format_patterns_indonesian(result: RuleBasedResult) -> str:
    """
    Generate a user-friendly Indonesian description of detected patterns.

    Args:
        result: Output from detect_patterns().

    Returns:
        Indonesian-language string describing the patterns found.
        Returns empty string if no patterns were detected.
    """
    if not result.patterns_found:
        return ""

    parts = []
    for category in result.categories_found:
        cat_display = CATEGORY_DISPLAY_ID.get(category, category)
        cat_labels = [m.label for m in result.patterns_found if m.category == category]
        labels_str = ", ".join(f"'{lbl}'" for lbl in cat_labels)
        parts.append(f"{cat_display} ({labels_str})")

    return "Pola linguistik yang terdeteksi: " + "; ".join(parts) + "."


# ── Module Self-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Pemerintah Indonesia mengumumkan kebijakan ekonomi baru untuk mendorong investasi.",
        "BREAKING!! Vaksin COVID-19 terbukti mengandung microchip 5G!! SEBARKAN sebelum dihapus!!",
        "AWAS!! BAHAYA!! Menurut ahli, makanan ini mengandung RACUN!! Kata pemerintah sudah terbukti!",
        "SEGERA bagikan pesan ini!! VIRAL!! Jangan sampai terhapus!! ANCAMAN besar!!",
        "Berita ini biasa saja, tidak ada pola mencurigakan.",
    ]

    for text in test_cases:
        result = detect_patterns(text)
        print(f"Text:       {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Score:      {result.support_score}")
        print(f"Categories: {result.categories_found}")
        print(f"Summary:    {result.summary}")
        if result.patterns_found:
            print(f"Patterns:   {get_pattern_labels(result)}")
            print(f"Display:    {format_patterns_indonesian(result)}")
        print()
