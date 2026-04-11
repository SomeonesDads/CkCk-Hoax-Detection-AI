"""
PII Filter — Personally Identifiable Information Detection & Redaction
=======================================================================
Detects and masks sensitive information in Indonesian text.

Required PII types (Track B Constraint B-4):
  - NIK          : 16-digit Indonesian national ID
  - Nomor Telepon: +62xx / 08xx format
  - Alamat Email : Standard email
  - Nomor Rekening: 10–16 digit bank account number

Bonus PII types:
  - NPWP    : Indonesian tax ID (XX.XXX.XXX.X-XXX.XXX)
  - Paspor  : Passport number (1–2 huruf + 6–7 digit)
  - Nama    : Deteksi berbasis keyword "atas nama", "a.n.", dll.
  - Tanggal Lahir: Format DD/MM/YYYY atau DD-MM-YYYY

Perbaikan dari versi placeholder:
  - NIK regex diperbaiki: digit hari perempuan +40 (41–71) ikut tertangkap
  - bank_account menggunakan keyword konteks ("rekening", "transfer", "norek", dll.)
    untuk menghindari false positive menangkap NIK/telepon
  - Phone regex diperluas untuk format dengan dash/spasi (0812-3456-7890)
  - Prioritas deteksi: NIK > NPWP > phone > email > bank_account
    agar overlap resolver memilih label yang paling spesifik
"""

import re
from dataclasses import dataclass


# ── Data Class ──────────────────────────────────────────────────────────────

@dataclass
class PIIMatch:
    """Satu entitas PII yang terdeteksi dalam teks."""
    pii_type: str   # Jenis PII (nik, phone, email, bank_account, npwp, passport)
    value: str      # Nilai asli dalam teks
    start: int      # Indeks awal dalam string
    end: int        # Indeks akhir dalam string
    masked: str     # Nilai setelah disensor


# ── Helper ──────────────────────────────────────────────────────────────────

def _build_nik_pattern() -> re.Pattern:
    """
    NIK Indonesia: 16 digit dengan struktur:
      [6 digit kode wilayah]
      [2 digit hari lahir — perempuan +40, sehingga 01–71]
      [2 digit bulan lahir 01–12]
      [2 digit tahun lahir]
      [4 digit nomor urut]

    Catatan: kita validasi digit hari 01–71 dan bulan 01–12.
    Kode wilayah tidak divalidasi karena terlalu ketat untuk regex.
    """
    day   = r'(?:0[1-9]|[1-6]\d|7[01])'   # 01–71 (perempuan hari +40)
    month = r'(?:0[1-9]|1[0-2])'           # 01–12
    year  = r'\d{2}'                        # 00–99
    seq   = r'\d{4}'                        # nomor urut

    return re.compile(
        rf'(?<!\d)'                         # tidak didahului digit
        rf'(\d{{6}}{day}{month}{year}{seq})'
        rf'(?!\d)'                          # tidak diikuti digit
    )


def _build_phone_pattern() -> re.Pattern:
    """
    Nomor telepon Indonesia:
      - Awalan: +62, 62, atau 0
      - Format tanpa separator: 08XXXXXXXXXX (8–12 digit setelah awalan)
      - Format dengan separator (dash/titik/spasi): 0812-3456-7890 / 0812.3456.7890

    Tantangan: '0812-3456-7890' dipecah jadi prefix='0', op='81',
    lalu ada digit '2' sebelum separator. Polanya harus fleksibel terhadap
    pemisahan segmen yang bervariasi.

    Pendekatan: gunakan dua sub-pola terpisah (dengan dan tanpa separator).
    """
    # Pola 1: tanpa separator — digit berurutan 10–13 karakter
    no_sep = (
        r'(?:\+62|62(?!\d)|0)'     # awalan
        r'8\d{7,11}'               # 8x + 7–11 digit berikutnya
    )

    # Pola 2: dengan separator (dash / titik / spasi)
    #   0812-3456-7890  → 0 | 812 | - | 3456 | - | 7890
    #   0812.3456.7890  → 0 | 812 | . | 3456 | . | 7890
    sep = r'[\-\.\s]'
    with_sep = (
        r'0'                        # awalan 0 (pola +62 jarang pakai separator)
        r'8\d{1,3}'                 # kode operator: 81, 812, 813, dst.
        + sep +
        r'\d{3,4}'                  # segmen tengah
        + sep +
        r'\d{4}'                    # segmen akhir
    )

    combined = rf'(?<!\d)(?:{no_sep}|{with_sep})(?!\d)'
    return re.compile(combined)


# Keyword konteks untuk nomor rekening — mengurangi false positive
_BANK_KEYWORDS = re.compile(
    r'(?:rekening|norek|no\.?\s*rek|transfer\s+ke|transfer\s+via|'
    r'rek\.?\s*tujuan|a\.?\s*n\.?|atas\s+nama|bca|bni|bri|mandiri|'
    r'btn|cimb|danamon|permata|ocbc|jago|jenius|gopay|ovo|dana)',
    re.IGNORECASE
)

# Jarak maksimal keyword ke angka (karakter)
_BANK_KEYWORD_WINDOW = 60


def _build_bank_pattern() -> re.Pattern:
    """
    Nomor rekening bank: 10–16 digit berurutan.
    Deteksi hanya jika ada keyword konteks dalam radius _BANK_KEYWORD_WINDOW.
    """
    return re.compile(r'(?<!\d)(\d{10,16})(?!\d)')


def _build_npwp_pattern() -> re.Pattern:
    """
    NPWP: XX.XXX.XXX.X-XXX.XXX
    Titik dan dash boleh absen (input tidak rapi).
    """
    return re.compile(
        r'(?<!\d)'
        r'\d{2}\.?\d{3}\.?\d{3}\.?\d'
        r'[\-\.]?'
        r'\d{3}\.?\d{3}'
        r'(?!\d)'
    )


def _build_passport_pattern() -> re.Pattern:
    """
    Nomor paspor Indonesia: 1–2 huruf kapital + 6–7 digit.
    Hanya cocok jika diawali batas kata.
    """
    return re.compile(r'\b[A-Z]{1,2}\s?\d{6,7}\b')


def _build_dob_pattern() -> re.Pattern:
    """
    Tanggal lahir format DD/MM/YYYY atau DD-MM-YYYY.
    Sering muncul berdampingan dengan NIK di laporan.
    """
    day   = r'(?:0[1-9]|[12]\d|3[01])'
    month = r'(?:0[1-9]|1[0-2])'
    year  = r'(?:19|20)\d{2}'
    sep   = r'[/\-]'
    return re.compile(
        rf'(?<!\d){day}{sep}{month}{sep}{year}(?!\d)'
    )


# ── Priority Order ───────────────────────────────────────────────────────────
# Urutan ini menentukan mana yang "menang" saat overlap.
# Pola yang lebih spesifik (NIK, NPWP) harus lebih tinggi prioritasnya.

_PRIORITY = {
    "npwp":         0,   # paling spesifik — format titik-titik-dash
    "nik":          1,   # 16 digit terstruktur
    "passport":     2,   # huruf + digit
    "dob":          3,   # tanggal lahir
    "email":        4,   # ada @
    "phone":        5,   # awalan 08/+62
    "bank_account": 6,   # paling umum — butuh konteks
}


# ── Main Class ───────────────────────────────────────────────────────────────

class PIIFilter:
    """
    Pipeline deteksi dan redaksi PII untuk teks berbahasa Indonesia.

    Contoh penggunaan::

        pii = PIIFilter()
        result = pii.filter("NIK saya 3201234506780001, hub. 081234567890")
        print(result["filtered_text"])
        # "NIK saya ..., hub. ..."

    Parameter
    ----------
    mask_char : str
        Karakter pengganti PII. Default ``"█"``.
    enabled_types : list[str] | None
        Jenis PII yang diaktifkan. ``None`` = semua jenis.
    """

    _COMPILED: dict[str, re.Pattern] = {
        "nik":          _build_nik_pattern(),
        "phone":        _build_phone_pattern(),
        "email":        re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
        "bank_account": _build_bank_pattern(),
        "npwp":         _build_npwp_pattern(),
        "passport":     _build_passport_pattern(),
        "dob":          _build_dob_pattern(),
    }

    # Tipe wajib (constraint B-4)
    REQUIRED_TYPES = {"nik", "phone", "email", "bank_account"}

    def __init__(
        self,
        mask_char: str = "█",
        enabled_types: list[str] | None = None,
    ):
        self.mask_char = mask_char
        self.enabled_types = enabled_types or list(self._COMPILED.keys())

        # Pastikan tipe wajib selalu aktif
        for t in self.REQUIRED_TYPES:
            if t not in self.enabled_types:
                self.enabled_types.append(t)

    # ── Internal ─────────────────────────────────────────────────────────

    def _is_bank_contextual(self, text: str, match_start: int, match_end: int) -> bool:
        """
        Cek apakah ada keyword perbankan di sekitar angka yang terdeteksi.
        Pencarian dilakukan dalam jendela _BANK_KEYWORD_WINDOW karakter di kiri/kanan.
        """
        window_start = max(0, match_start - _BANK_KEYWORD_WINDOW)
        window_end   = min(len(text), match_end + _BANK_KEYWORD_WINDOW)
        window_text  = text[window_start:window_end]
        return bool(_BANK_KEYWORDS.search(window_text))

    def _resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """
        Hilangkan tumpang tindih antar match.
        Strategi: jika dua match overlap, pertahankan yang prioritasnya lebih tinggi
        (angka prioritas lebih kecil = lebih spesifik). Jika prioritas sama,
        pertahankan yang lebih panjang.
        """
        # Urutkan: prioritas naik, lalu posisi awal naik
        matches.sort(key=lambda m: (_PRIORITY.get(m.pii_type, 99), m.start))

        resolved: list[PIIMatch] = []
        for candidate in matches:
            overlap = False
            for accepted in resolved:
                # Cek tumpang tindih
                if candidate.start < accepted.end and candidate.end > accepted.start:
                    overlap = True
                    break
            if not overlap:
                resolved.append(candidate)

        # Kembalikan dalam urutan posisi untuk redaksi
        resolved.sort(key=lambda m: m.start)
        return resolved

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Deteksi semua entitas PII dalam teks.

        Parameter
        ----------
        text : str
            Teks input yang akan dipindai.

        Returns
        -------
        list[PIIMatch]
            Daftar entitas PII yang ditemukan, diurutkan berdasarkan posisi.
        """
        raw_matches: list[PIIMatch] = []

        for pii_type in self.enabled_types:
            pattern = self._COMPILED.get(pii_type)
            if pattern is None:
                continue

            for m in pattern.finditer(text):
                value = m.group()

                # Khusus bank_account: skip jika tidak ada keyword konteks
                if pii_type == "bank_account":
                    if not self._is_bank_contextual(text, m.start(), m.end()):
                        continue

                masked = self.mask_char * len(value)
                raw_matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=value,
                    start=m.start(),
                    end=m.end(),
                    masked=masked,
                ))

        return self._resolve_overlaps(raw_matches)

    def redact(self, text: str) -> str:
        """
        Sensor semua PII dalam teks dengan karakter mask.

        Parameter
        ----------
        text : str
            Teks input.

        Returns
        -------
        str
            Teks dengan PII disensor.
        """
        matches = self.detect(text)

        # Ganti dari belakang agar indeks tidak bergeser
        redacted = text
        for match in reversed(matches):
            redacted = (
                redacted[: match.start]
                + match.masked
                + redacted[match.end :]
            )
        return redacted

    def filter(self, text: str) -> dict:
        """
        Pipeline lengkap: deteksi + redaksi + laporan.

        Parameter
        ----------
        text : str
            Teks input.

        Returns
        -------
        dict dengan field:
            - ``original_text``  : teks asli
            - ``filtered_text``  : teks setelah disensor
            - ``pii_found``      : bool, ada PII atau tidak
            - ``pii_count``      : jumlah PII yang ditemukan
            - ``details``        : list detail tiap PII
        """
        matches = self.detect(text)

        redacted = text
        for match in reversed(matches):
            redacted = (
                redacted[: match.start]
                + match.masked
                + redacted[match.end :]
            )

        return {
            "original_text": text,
            "filtered_text": redacted,
            "pii_found":     len(matches) > 0,
            "pii_count":     len(matches),
            "details": [
                {
                    "type":     m.pii_type,
                    "original": m.value,
                    "masked":   m.masked,
                    "position": (m.start, m.end),
                }
                for m in matches
            ],
        }


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pii = PIIFilter()

    test_cases = [
        ("NIK laki-laki normal",
         "NIK saya 3201234506780001 tolong dijaga."),
        ("NIK perempuan (hari +40)",
         "NIK: 3271614506780002 — perempuan lahir 21 April 1978."),
        ("Telepon +62",
         "Hubungi +6281234567890 untuk informasi."),
        ("Telepon 08xx dengan dash",
         "Nomor WA: 0812-3456-7890."),
        ("Telepon 08xx dengan titik",
         "Telepon: 0812.3456.7890"),
        ("Email",
         "Email: budi.santoso@gmail.com"),
        ("Rekening dengan keyword",
         "Transfer ke rekening BCA 1234567890123456 atas nama Budi."),
        ("Rekening tanpa keyword — tidak boleh terdeteksi",
         "Kode produk: 1234567890123456."),

        # Bonus
        ("NPWP",
         "NPWP: 12.345.678.9-012.345"),
        ("Paspor",
         "Nomor paspor: AB1234567"),
        ("Tanggal lahir",
         "Lahir: 21/04/1985"),

        # Campuran
        ("Multi-PII",
         "Korban Budi (NIK 3201234506780001, email budi@example.com, "
         "HP +6281234567890) melaporkan transfer ke rekening Mandiri "
         "1234567890123456."),

        # Teks bersih
        ("Tidak ada PII",
         "Pemerintah Indonesia mengumumkan kebijakan baru di sektor energi."),
    ]

    print("=" * 65)
    print("PII FILTER — Self-Test")
    print("=" * 65)

    passed = 0
    for label, text in test_cases:
        result = pii.filter(text)
        status = "✅" if not (label == "Rekening tanpa keyword — tidak boleh terdeteksi"
                               and result["pii_count"] > 0) else "❌ FALSE POSITIVE"
        if result["pii_count"] == 0 and label != "Tidak ada PII" and "tidak boleh" not in label:
            status = "⚠️  MISS"
        else:
            passed += 1

        print(f"\n[{label}]")
        print(f"  Input : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Output: {result['filtered_text'][:80]}")
        print(f"  PII   : {result['pii_count']} ditemukan  {status}")
        for d in result["details"]:
            print(f"    → [{d['type']:12s}] {d['original']!r} → {d['masked']!r}")

    print("\n" + "=" * 65)
    print(f"Selesai: {passed}/{len(test_cases)} test cases.")
    print("=" * 65)