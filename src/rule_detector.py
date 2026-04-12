"""
Rule-based Detector
======================
Detects manipulative linguistic patterns in Indonesian text.
Provides explainability layer — explains WHY content is suspicious,
without changing the classifier's confidence score.

Categories:
  1. Urgensi Palsu (False Urgency)
  2. Fear-mongering
  3. Atribusi Invalid (Invalid Attribution)
"""

import re
from dataclasses import dataclass, field


@dataclass
class PatternMatch:
    """Represents a detected manipulative pattern."""
    category: str          # urgency, fear, attribution
    pattern: str           # The matched keyword/phrase
    context: str           # Surrounding text snippet
    description: str       # Human-readable explanation (Bahasa Indonesia)


class RuleBasedDetector:
    """
    Detects manipulative linguistic patterns commonly found in Indonesian hoaxes.
    
    This component is an EXPLAINABILITY layer — it does NOT change the model's
    confidence score. It only enriches the output with reasons.
    
    Usage:
        detector = RuleBasedDetector()
        patterns = detector.detect("SEGERA SEBARKAN!! Bahaya vaksin terbukti!!")
        for p in patterns:
            print(f"[{p.category}] '{p.pattern}' — {p.description}")
    """

    # ── Pattern Definitions ─────────────────────────────────────────────

    URGENCY_PATTERNS = {
        r'\bSEGERA\b': 'Menggunakan kata "SEGERA" untuk menciptakan tekanan waktu',
        r'\bVIRAL\b': 'Mengklaim konten "VIRAL" untuk mendorong penyebaran',
        r'\bSEBARKAN\b': 'Ajakan menyebarkan — pola umum hoaks berantai',
        r'\bBAGIKAN\b': 'Ajakan membagikan — pola umum hoaks berantai',
        r'\bJANGAN\s+SAMPAI\s+TERHAPUS\b': 'Klaim konten akan dihapus — taktik urgensi palsu',
        r'\bSEBELUM\s+DIHAPUS\b': 'Klaim konten akan dihapus — taktik urgensi palsu',
        r'\bBREAKING\b': 'Menggunakan "BREAKING" untuk kesan berita mendesak',
        r'\bDARURAT\b': 'Menggunakan kata "DARURAT" untuk menciptakan kepanikan',
        r'\b(?:WAJIB|HARUS)\s+(?:BACA|TONTON|LIHAT|SHARE)\b': 'Perintah agresif untuk menyebarkan konten',
        r'[!]{3,}': 'Penggunaan tanda seru berlebihan (!!!) — indikator emosi manipulatif',
        r'[A-Z\s]{15,}': 'Penggunaan CAPSLOCK berlebihan — indikator sensasionalisme',
    }

    FEAR_PATTERNS = {
        r'\bBAHAYA\b': 'Menggunakan kata "BAHAYA" untuk memicu ketakutan',
        r'\bANCAMAN\b': 'Menggunakan kata "ANCAMAN" untuk memicu ketakutan',
        r'\bKORBAN\b': 'Menyebut "KORBAN" untuk membangun narasi ketakutan',
        r'\bHANCUR\b': 'Menggunakan kata "HANCUR" untuk dramatisasi',
        r'\bMATI\b': 'Menggunakan kata "MATI" untuk menciptakan rasa takut',
        r'\bRACUN\b': 'Klaim "RACUN" — umum dalam hoaks kesehatan',
        r'\bBERBHAYA\b': 'Menggunakan kata "BERBAHAYA" untuk memicu ketakutan',
        r'\bMENGERIKAN\b': 'Menggunakan kata "MENGERIKAN" untuk dramatisasi',
        r'\bMENYERAMKAN\b': 'Menggunakan kata "MENYERAMKAN" untuk dramatisasi',
        r'\bAWAS\b': 'Kata peringatan "AWAS" — pola umum hoaks',
        r'\bWASPADA\b': 'Kata peringatan "WASPADA" — sering digunakan untuk memicu kecemasan',
    }

    ATTRIBUTION_PATTERNS = {
        r'(?:menurut|kata)\s+(?:ahli|pakar|dokter|profesor|ilmuwan)\b': 
            'Klaim kutipan ahli tanpa nama spesifik — atribusi tidak terverifikasi',
        r'(?:menurut|kata)\s+(?:sumber|pihak)\s+(?:terpercaya|berwenang)\b': 
            'Klaim "sumber terpercaya" tanpa identitas — atribusi invalid',
        r'\btelah\s+terbukti\b': 
            'Klaim "telah terbukti" tanpa referensi — pernyataan tidak berdasar',
        r'\bsecara\s+ilmiah\b': 
            'Klaim "secara ilmiah" tanpa sitasi jurnal — pseudosains',
        r'\bpenelitian\s+(?:menunjukkan|membuktikan)\b': 
            'Klaim hasil penelitian tanpa referensi spesifik',
        r'\b(?:WHO|PBB|pemerintah)\s+(?:menyatakan|mengumumkan|mengakui)\b': 
            'Klaim pernyataan institusi — perlu verifikasi sumber resmi',
        r'\bfakta\s+yang\s+(?:disembunyikan|ditutupi)\b':
            'Narasi konspirasi — klaim informasi disembunyikan',
    }

    CATEGORY_MAP = {
        "urgency": ("Urgensi Palsu", URGENCY_PATTERNS),
        "fear": ("Fear-mongering", FEAR_PATTERNS),
        "attribution": ("Atribusi Invalid", ATTRIBUTION_PATTERNS),
    }

    def __init__(self, enabled_categories: list[str] | None = None):
        """
        Initialize Rule-based Detector.
        
        Args:
            enabled_categories: List of categories to detect.
                Options: 'urgency', 'fear', 'attribution'
                None = all categories.
        """
        self.enabled_categories = enabled_categories or list(self.CATEGORY_MAP.keys())

    def detect(self, text: str) -> list[PatternMatch]:
        """
        Detect all manipulative patterns in the text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of PatternMatch objects describing detected patterns.
        """
        matches = []

        for category_key in self.enabled_categories:
            if category_key not in self.CATEGORY_MAP:
                continue

            category_name, patterns = self.CATEGORY_MAP[category_key]

            for pattern_str, description in patterns.items():
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for match in pattern.finditer(text):
                    # Extract context (30 chars before and after)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    if start > 0:
                        context = "..." + context
                    if end < len(text):
                        context = context + "..."

                    matches.append(PatternMatch(
                        category=category_name,
                        pattern=match.group(),
                        context=context,
                        description=description,
                    ))

        return matches

    def get_summary(self, text: str) -> dict:
        """
        Get a structured summary of all detected patterns.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dict with pattern counts, categories found, and detailed matches.
        """
        matches = self.detect(text)

        # Group by category
        by_category = {}
        for m in matches:
            if m.category not in by_category:
                by_category[m.category] = []
            by_category[m.category].append({
                "pattern": m.pattern,
                "description": m.description,
                "context": m.context,
            })

        return {
            "total_patterns": len(matches),
            "categories_found": list(by_category.keys()),
            "has_manipulative_patterns": len(matches) > 0,
            "details": by_category,
        }


if __name__ == "__main__":
    detector = RuleBasedDetector()

    test_cases = [
        "SEGERA SEBARKAN!! Vaksin COVID-19 terbukti mengandung RACUN!! Menurut ahli kesehatan, ini BAHAYA besar!!!",
        "Pemerintah Indonesia mengumumkan kebijakan ekonomi baru untuk mendorong pertumbuhan investasi.",
        "AWAS!! JANGAN SAMPAI TERHAPUS!! Penelitian menunjukkan makanan ini BERBAHAYA!! VIRAL!!",
        "Presiden menyampaikan pidato kenegaraan di Istana Merdeka pada Senin pagi.",
    ]

    for text in test_cases:
        print(f"Text: {text[:80]}...")
        summary = detector.get_summary(text)
        print(f"  Patterns found: {summary['total_patterns']}")
        if summary["has_manipulative_patterns"]:
            for cat, details in summary["details"].items():
                print(f"  [{cat}]:")
                for d in details:
                    print(f"    → '{d['pattern']}' — {d['description']}")
        else:
            print("  ✅ No manipulative patterns detected")
        print()
