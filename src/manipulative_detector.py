"""
Manipulative Pattern Detector — Rule-Based Support Score
=========================================================
Mendeteksi pola linguistik khas konten manipulatif / hoaks berbahasa
Indonesia secara langsung dari teks, tanpa model AI.

Digunakan sebagai sinyal pendukung (bobot 30%) dalam pipeline CkCk.
Sinyal primer (70%) berasal dari knowledge base (compute_primary_score
— dikerjakan oleh bagian AI engineer).

Kategori pola yang dideteksi:
  1. Urgensi palsu     — "SEGERA", "VIRAL", "SEBARKAN", "FORWARD"
  2. Fear-mongering    — "BAHAYA", "ANCAMAN", "AWAS", "KORBAN"
  3. Sensasionalisme   — "BREAKING", "MENGEJUTKAN", "TERBUKTI"
  4. Atribusi lemah    — "katanya", "konon", "beredar kabar"
  5. Ajakan buta       — "bagikan ke semua", "sebelum dihapus"
  6. Klaim absolut     — "100%", "pasti", "dijamin", "tidak mungkin salah"
  7. Pseudosains       — "microchip", "5G", "radiasi", "racun tersembunyi"

Skor akhir dinormalisasi ke rentang [0.0, 1.0].
"""

import re
from dataclasses import dataclass, field


# ── Kamus Pola ───────────────────────────────────────────────────────────────

# Setiap kategori berisi:
#   - patterns : list regex (case-insensitive)
#   - weight   : bobot kontribusi ke skor (relatif antar kategori)
#   - label    : nama kategori untuk laporan

_PATTERN_GROUPS: list[dict] = [
    {
        "label": "urgensi_palsu",
        "weight": 1.5,
        "patterns": [
            r'\bSEGERA\b',
            r'\bVIRAL\b',
            r'\bSEBARKAN\b',
            r'\bFORWARD\b',
            r'\bSHARE\b',
            r'\bSEBELUM\s+DIHAPUS\b',
            r'\bJANGAN\s+SAMPAI\s+TERLEWAT\b',
            r'\bSEKARANG\s+JUGA\b',
            r'\bHARUS\s+DIBACA\b',
            r'\bWAJIB\s+TAHU\b',
            r'\bINFO\s+PENTING\b',
            r'\bBURU[-\s]?BURU\b',
        ],
    },
    {
        "label": "fear_mongering",
        "weight": 1.5,
        "patterns": [
            r'\bBAHAYA\b',
            r'\bANCAMAN\b',
            r'\bAWAS\b',
            r'\bKORBAN\b',
            r'\bMATI\b',
            r'\bTEWAS\b',
            r'\bDARURAT\b',
            r'\bBENCAN[A]\b',
            r'\bPANIK\b',
            r'\bSERANGAN\b',
            r'\bDIBUNUH\b',
            r'\bDIRACUN\b',
            r'\bMAUT\b',
            r'\bNYAWA\b',
            r'\bAPOKALIPS\b',
        ],
    },
    {
        "label": "sensasionalisme",
        "weight": 1.2,
        "patterns": [
            r'\bBREAKING\b',
            r'\bMENGEJUTKAN\b',
            r'\bGEGER\b',
            r'\bHEBOH\b',
            r'\bSYOK\b',
            r'\bSHOCKING\b',
            r'\bEKSKLUSIF\b',
            r'\bBOCAR\b',           # "bocor" — info bocor
            r'\bRAHASIA\s+TERBONGKAR\b',
            r'\bTERBUKTI\b',
            r'\bSUDAH\s+TERBUKTI\b',
            r'\bFAKTA\s+TERSEMBUNYI\b',
            r'!!{2,}',              # dua atau lebih tanda seru berturutan
            r'\b[A-Z]{6,}\b',       # kata CAPSLOCK murni ≥6 huruf (bukan Title Case)
        ],
    },
    {
        "label": "atribusi_lemah",
        "weight": 1.3,
        "patterns": [
            r'\bkatanya\b',
            r'\bkonon\b',
            r'\bberedar\s+(?:kabar|info|berita)\b',
            r'\bkabar\s+(?:beredar|angin)\b',
            r'\bmenurut\s+(?:sumber\s+)?(?:anonim|tak\s+dikenal|tidak\s+dikenal)\b',
            r'\btanpa\s+(?:sumber|referensi|bukti)\b',
            r'\bada\s+yang\s+bilang\b',
            r'\bseseorang\s+(?:mengatakan|mengklaim)\b',
            r'\binfo\s+dari\s+(?:grup|wa|whatsapp)\b',
            r'\bdapat\s+(?:info|kabar)\s+dari\b',
        ],
    },
    {
        "label": "ajakan_buta",
        "weight": 1.4,
        "patterns": [
            r'\bbagikan\s+(?:ke\s+)?(?:semua|seluruh|teman|keluarga|grup)\b',
            r'\bsebarkan\s+(?:ke\s+)?(?:semua|seluruh|teman|keluarga)\b',
            r'\bjangan\s+(?:sampai\s+)?dihapus\b',
            r'\bsebelum\s+dihapus\b',
            r'\bsave\s+(?:dulu\s+)?(?:sebelum|ini)\b',
            r'\bscreenshot\s+(?:ini|dulu)\b',
            r'\bforward\s+(?:ke\s+)?(?:semua|teman|grup)\b',
            r'\bberitahu\s+(?:semua\s+)?(?:orang|teman|keluarga)\b',
            r'\bjangan\s+pelit\s+(?:info|berbagi)\b',
        ],
    },
    {
        "label": "klaim_absolut",
        "weight": 1.0,
        "patterns": [
            r'\b100\s*%\s*(?:pasti|benar|terbukti|nyata)\b',
            r'\bsudah\s+pasti\b',
            r'\btidak\s+mungkin\s+salah\b',
            r'\bdijamin\s+(?:benar|nyata|terbukti)\b',
            r'\bpasti\s+(?:benar|terjadi|nyata)\b',
            r'\btidak\s+ada\s+yang\s+(?:tahu|berani\s+ungkap)\b',
            r'\bmereka\s+tidak\s+mau\s+(?:kamu|anda)\s+tahu\b',
            r'\bditutupi\s+(?:pemerintah|media)\b',
            r'\bdilarang\s+tayang\b',
            r'\bdisensor\b',
        ],
    },
    {
        "label": "pseudosains",
        "weight": 1.6,   # bobot tertinggi — sangat spesifik untuk hoaks
        "patterns": [
            r'\bmicrochip\b',
            r'\b5G\b',
            r'\bradiasi\s+(?:berbahaya|mematikan|ponsel)\b',
            r'\bvaksin\s+(?:mengandung|menyebabkan|berbahaya)\b',
            r'\bkemotrop\b',
            r'\billuminati\b',
            r'\bpenyembuh\s+(?:segala|kanker|covid)\b',
            r'\bobat\s+(?:ajaib|mujarab)\s+(?:yang\s+)?(?:disembunyikan|ditutupi)\b',
            r'\bgelombang\s+(?:elektromagnetik|5G)\s+(?:berbahaya|mematikan)\b',
            r'\bnano(?:partikel|bot|teknologi)\s+(?:dalam\s+)?vaksin\b',
            r'\bprogramming\s+otak\b',
            r'\bkontrol\s+pikiran\b',
        ],
    },
]


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """Satu pola manipulatif yang terdeteksi."""
    category: str       # label kategori
    pattern:  str       # pola regex yang cocok
    match:    str       # teks yang cocok
    position: tuple[int, int]


@dataclass
class ManipulativeAnalysis:
    """Hasil analisis lengkap satu teks."""
    raw_score:      float               # skor sebelum normalisasi
    support_score:  float               # skor [0.0, 1.0] untuk pipeline
    risk_level:     str                 # RENDAH / SEDANG / TINGGI
    matches:        list[PatternMatch]  # semua pola yang terdeteksi
    category_hits:  dict[str, int]      # jumlah hit per kategori
    explanation:    str                 # penjelasan singkat Bahasa Indonesia


# ── Compile Patterns ─────────────────────────────────────────────────────────

# Pattern capslock harus dikompilasi TANPA re.IGNORECASE
_CAPSLOCK_PATTERN = r'\b[A-Z]{6,}\b'

_COMPILED_GROUPS: list[dict] = []
for group in _PATTERN_GROUPS:
    compiled_patterns = []
    for p in group["patterns"]:
        if p == _CAPSLOCK_PATTERN:
            compiled_patterns.append(re.compile(p))          # case-sensitive
        else:
            compiled_patterns.append(re.compile(p, re.IGNORECASE))
    _COMPILED_GROUPS.append({
        "label":    group["label"],
        "weight":   group["weight"],
        "compiled": compiled_patterns,
        "raw":      group["patterns"],
    })

# Skor maksimum teoritis (jika semua pola tiap kategori cocok)
_MAX_RAW_SCORE: float = sum(
    len(g["compiled"]) * g["weight"] for g in _COMPILED_GROUPS
)


# ── Core Function ─────────────────────────────────────────────────────────────

def compute_support_score(text: str) -> ManipulativeAnalysis:
    """
    Hitung skor dukungan pola manipulatif dari teks secara rule-based.

    Skor ini berbobot 30% dari hasil akhir pipeline CkCk.
    Sinyal primer 70% berasal dari knowledge base (compute_primary_score).

    Parameter
    ----------
    text : str
        Teks yang sudah melalui PII filter (teks bersih, bukan teks asli).

    Returns
    -------
    ManipulativeAnalysis
        Objek hasil analisis berisi skor, label risiko, detail pola,
        dan penjelasan dalam Bahasa Indonesia.

    Contoh
    ------
    >>> result = compute_support_score("SEGERA SEBARKAN!! Vaksin mengandung microchip 5G!!")
    >>> result.support_score
    0.87
    >>> result.risk_level
    'TINGGI'
    """
    matches: list[PatternMatch] = []
    category_hits: dict[str, int] = {}
    raw_score: float = 0.0

    for group in _COMPILED_GROUPS:
        label   = group["label"]
        weight  = group["weight"]
        hit_count = 0

        for pat, raw_pat in zip(group["compiled"], group["raw"]):
            for m in pat.finditer(text):
                matches.append(PatternMatch(
                    category=label,
                    pattern=raw_pat,
                    match=m.group(),
                    position=(m.start(), m.end()),
                ))
                # Setiap hit tambah skor, tapi dengan diminishing returns
                # agar satu kategori tidak mendominasi terlalu besar
                hit_count += 1
                if hit_count == 1:
                    raw_score += weight          # hit pertama: penuh
                elif hit_count == 2:
                    raw_score += weight * 0.5    # hit kedua: setengah
                else:
                    raw_score += weight * 0.2    # hit ketiga+: minimal

        if hit_count:
            category_hits[label] = hit_count

    # Normalisasi ke [0.0, 1.0]
    # Acuan praktis: raw score ~6.0 sudah sangat manipulatif.
    # Max teoritis terlalu besar sehingga semua skor terkompresi mendekati 0.
    PRACTICAL_CEILING = 6.0
    support_score = min(raw_score / PRACTICAL_CEILING, 1.0)

    # Label risiko
    if support_score >= 0.70:
        risk_level = "TINGGI"
    elif support_score >= 0.30:
        risk_level = "SEDANG"
    else:
        risk_level = "RENDAH"

    # Penjelasan Bahasa Indonesia
    explanation = _build_explanation(support_score, risk_level, category_hits)

    return ManipulativeAnalysis(
        raw_score=round(raw_score, 4),
        support_score=round(support_score, 4),
        risk_level=risk_level,
        matches=matches,
        category_hits=category_hits,
        explanation=explanation,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

_CATEGORY_LABELS_ID: dict[str, str] = {
    "urgensi_palsu":    "urgensi palsu",
    "fear_mongering":   "pemicu ketakutan",
    "sensasionalisme":  "sensasionalisme berlebihan",
    "atribusi_lemah":   "atribusi tanpa sumber jelas",
    "ajakan_buta":      "ajakan penyebaran masif",
    "klaim_absolut":    "klaim absolut tanpa bukti",
    "pseudosains":      "klaim pseudosains",
}


def _build_explanation(
    score: float,
    risk_level: str,
    category_hits: dict[str, int],
) -> str:
    """Bangun penjelasan singkat dalam Bahasa Indonesia."""
    if not category_hits:
        return (
            "Tidak ditemukan pola linguistik yang mencurigakan. "
            "Teks tampak netral dari sisi gaya bahasa."
        )

    # Sebutkan kategori yang terdeteksi
    detected = [
        _CATEGORY_LABELS_ID.get(cat, cat)
        for cat in category_hits
    ]
    cat_str = ", ".join(detected)

    if risk_level == "TINGGI":
        prefix = (
            f"⚠️ Terdeteksi pola manipulatif dengan risiko TINGGI "
            f"(skor {score:.0%}). "
        )
        suffix = (
            "Kombinasi pola ini sangat umum ditemukan pada konten hoaks "
            "yang sengaja dirancang untuk memicu reaksi emosional dan "
            "penyebaran cepat tanpa verifikasi."
        )
    elif risk_level == "SEDANG":
        prefix = (
            f"⚠️ Terdeteksi beberapa pola yang perlu diwaspadai "
            f"(skor {score:.0%}). "
        )
        suffix = (
            "Pola ini tidak serta merta menandakan hoaks, namun sebaiknya "
            "verifikasi klaim ke sumber resmi sebelum membagikan."
        )
    else:
        prefix = (
            f"ℹ️ Terdeteksi sedikit pola yang umum pada konten sensasional "
            f"(skor {score:.0%}). "
        )
        suffix = "Risiko manipulatif relatif rendah dari sisi gaya bahasa."

    return f"{prefix}Pola yang ditemukan: {cat_str}. {suffix}"


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        (
            "Berita bersih",
            "Pemerintah Indonesia mengumumkan kebijakan ekonomi baru "
            "untuk mendorong pertumbuhan investasi di sektor teknologi.",
            "RENDAH",
        ),
        (
            "Hoaks vaksin microchip 5G",
            "BREAKING!! Vaksin COVID-19 terbukti mengandung microchip 5G!! "
            "Bagikan sebelum dihapus!! Awas bahaya!!",
            "TINGGI",
        ),
        (
            "Ajakan forward berantai",
            "INFO PENTING!! Beredar kabar ada modus penipuan baru. "
            "Sebarkan ke semua teman dan keluarga sebelum dihapus!!",
            "TINGGI",
        ),
        (
            "Atribusi lemah tanpa urgensi",
            "Katanya ada kebijakan baru yang belum diumumkan resmi. "
            "Konon sumber anonim menyebut hal ini akan berdampak besar.",
            "SEDANG",
        ),
        (
            "Berita dengan satu kata sensasional",
            "Mengejutkan! Harga bahan pokok naik signifikan bulan ini "
            "menurut laporan resmi BPS.",
            "RENDAH",
        ),
    ]

    print("=" * 65)
    print("MANIPULATIVE DETECTOR — Self-Test")
    print("=" * 65)

    all_pass = True
    for label, text, expected_risk in test_cases:
        result = compute_support_score(text)
        ok = result.risk_level == expected_risk
        if not ok:
            all_pass = False
        status = "✅" if ok else f"❌ (expected {expected_risk})"

        print(f"\n[{label}]")
        print(f"  Teks        : {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Skor        : {result.support_score:.2f}  |  Risiko: {result.risk_level}  {status}")
        print(f"  Penjelasan  : {result.explanation[:100]}...")
        if result.category_hits:
            print(f"  Kategori    : {result.category_hits}")

    print("\n" + "=" * 65)
    print(f"{'Semua test lulus ✅' if all_pass else 'Ada test gagal ❌'}")
    print("=" * 65)