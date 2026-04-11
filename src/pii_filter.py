"""
PII Filter — Personally Identifiable Information Detection & Redaction
=======================================================================
Detects and masks sensitive information in Indonesian text.

Required PII types (Track B Constraint B-4):
  - NIK (16-digit Indonesian national ID)
  - Nomor Telepon (+62xx / 08xx format)
  - Alamat Email
  - Nomor Rekening (16-digit bank account)

Bonus PII types:
  - NPWP (Indonesian tax ID)
  - Nomor Paspor (Passport number)
"""

import re
from dataclasses import dataclass, field


@dataclass
class PIIMatch:
    """Represents a detected PII entity."""
    pii_type: str
    value: str
    start: int
    end: int
    masked: str


class PIIFilter:
    """
    PII detection and redaction pipeline for Indonesian text.
    
    Usage:
        pii_filter = PIIFilter()
        result = pii_filter.filter("Hubungi saya di 081234567890")
        print(result["filtered_text"])
        # Output: "Hubungi saya di ████████████"
    """

    # ── Regex Patterns ──────────────────────────────────────────────────
    PATTERNS = {
        # NIK: exactly 16 digits (Indonesian national ID)
        # Day field: 01-31 for male, 41-71 for female (day + 40)
        "nik": re.compile(
            r'\b\d{6}(?:0[1-9]|[12]\d|3[01]|4[1-9]|[56]\d|7[01])(?:0[1-9]|1[0-2])\d{2}\d{4}\b'
        ),

        # Phone: Indonesian format (+62 / 62 / 08xx)
        "phone": re.compile(
            r'(?:\+62|62|0)[\s\-]?'           # prefix
            r'(?:8\d{1}[\s\-]?\d{3,4}[\s\-]?\d{3,4}'  # mobile: 08xx
            r'|(?:21|22|24|31|61|71|411|421|451|471|541|551|561|711)\s?\d{5,8})'  # landline
        ),

        # Email
        "email": re.compile(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
        ),

        # Bank account: 10-16 digit number (commonly 16 digits)
        "bank_account": re.compile(
            r'\b\d{10,16}\b'
        ),

        # NPWP: format XX.XXX.XXX.X-XXX.XXX (Indonesian tax ID) — Bonus
        "npwp": re.compile(
            r'\b\d{2}\.?\d{3}\.?\d{3}\.?\d[\-.]?\d{3}\.?\d{3}\b'
        ),

        # Passport: 1-2 letters followed by 6-7 digits — Bonus
        "passport": re.compile(
            r'\b[A-Z]{1,2}\s?\d{6,7}\b'
        ),
    }

    def __init__(self, mask_char: str = "█", enabled_types: list[str] | None = None):
        """
        Initialize PII Filter.
        
        Args:
            mask_char: Character used for masking PII.
            enabled_types: List of PII types to detect. None = all types.
        """
        self.mask_char = mask_char
        self.enabled_types = enabled_types or list(self.PATTERNS.keys())

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Detect all PII entities in the text.
        
        Args:
            text: Input text to scan.
            
        Returns:
            List of PIIMatch objects.
        """
        matches = []

        for pii_type in self.enabled_types:
            if pii_type not in self.PATTERNS:
                continue

            pattern = self.PATTERNS[pii_type]
            for match in pattern.finditer(text):
                value = match.group()
                masked = self.mask_char * len(value)
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    masked=masked,
                ))

        # Sort by position (start index)
        matches.sort(key=lambda m: m.start)

        # Remove overlapping matches (keep the longest)
        filtered = []
        for match in matches:
            if filtered and match.start < filtered[-1].end:
                # Overlapping — keep the longer match
                if (match.end - match.start) > (filtered[-1].end - filtered[-1].start):
                    filtered[-1] = match
            else:
                filtered.append(match)

        return filtered

    def redact(self, text: str) -> str:
        """
        Redact all PII from text, replacing with mask characters.
        
        Args:
            text: Input text.
            
        Returns:
            Text with PII redacted.
        """
        matches = self.detect(text)

        # Replace from end to start to preserve indices
        redacted = text
        for match in reversed(matches):
            redacted = redacted[:match.start] + match.masked + redacted[match.end:]

        return redacted

    def filter(self, text: str) -> dict:
        """
        Full PII filtering pipeline.
        
        Args:
            text: Input text.
            
        Returns:
            Dict with filtered_text, original_text, pii_found, and details.
        """
        matches = self.detect(text)
        filtered_text = self.redact(text)

        return {
            "original_text": text,
            "filtered_text": filtered_text,
            "pii_found": len(matches) > 0,
            "pii_count": len(matches),
            "details": [
                {
                    "type": m.pii_type,
                    "original": m.value,
                    "masked": m.masked,
                    "position": (m.start, m.end),
                }
                for m in matches
            ],
        }


if __name__ == "__main__":
    # ── Demo ────────────────────────────────────────────────────────────
    pii = PIIFilter()

    test_cases = [
        "NIK saya 3201234506780001 tolong dijaga.",
        "Hubungi saya di +6281234567890 atau 08119876543.",
        "Email: budi.santoso@gmail.com",
        "Transfer ke rekening 1234567890123456.",
        "NPWP: 12.345.678.9-012.345",
        "Paspor: AB1234567",
        "Berita ini tidak mengandung data pribadi apapun.",
    ]

    for text in test_cases:
        result = pii.filter(text)
        print(f"Input:  {result['original_text']}")
        print(f"Output: {result['filtered_text']}")
        print(f"PII:    {result['pii_count']} found")
        if result["details"]:
            for d in result["details"]:
                print(f"  → [{d['type']}] {d['original']} → {d['masked']}")
        print()
