# BAB III: Kepatuhan Constraint — Track B: The Privacy Brain

> Penjelasan bagaimana model memenuhi setiap constraint Track B.

---

## Constraint 1: Model Size ≤ 4 Miliar Parameter

**Status**: ✅ COMPLIANT

**Model yang digunakan**: `indobenchmark/indobert-base-p2`

| Metric | Value |
|---|---|
| Total Parameters | ~110,000,000 (~110M) |
| Batas Maksimal | 4,000,000,000 (4B) |
| Rasio Penggunaan | ~2.75% dari batas |

IndoBERT-base-p2 adalah Small Language Model (SLM) berbasis arsitektur BERT-base yang telah di-pretrain pada korpus bahasa Indonesia. Dengan hanya ~110M parameter, model ini jauh di bawah batas 4B dan dapat berjalan efisien pada CPU.

**Verifikasi**:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: ~110,000,000
```

---

## Constraint 2: Offline Total (Zero Network Call)

**Status**: ✅ COMPLIANT

Seluruh pipeline inferensi berjalan 100% secara lokal tanpa network call:

1. **Model weights** di-download sekali saat setup, kemudian disimpan di folder `models/`
2. **Tidak ada API call** ke server eksternal dalam `inference.ipynb`
3. **Tidak ada dependency** pada layanan cloud (OpenAI, Gemini, Claude, HuggingFace Inference API, dll)

**Mekanisme Offline**:
- Model dimuat dari checkpoint lokal (`models/best_model/`)
- Tokenizer disimpan bersama model
- PII Filter menggunakan regex (tidak memerlukan koneksi internet)
- Preprocessing sepenuhnya rule-based

**Verifikasi**: Jalankan `inference.ipynb` dengan koneksi internet dimatikan.

---

## Constraint 3: PII Filter Wajib

**Status**: ✅ COMPLIANT

PII Filter terintegrasi langsung dalam pipeline inferensi (`inference.ipynb`), bukan sebagai skrip terpisah.

**Implementasi**: `src/pii_filter.py` — Class `PIIFilter`

**Pipeline Flow**:
```
Input Text → PII Filter (detect + redact) → Preprocessor → Model → Output
```

PII Filter berjalan **sebelum** teks diproses oleh model, memastikan data sensitif tidak pernah masuk ke model.

---

## Constraint 4: Cakupan PII

**Status**: ✅ COMPLIANT (+ Bonus)

### PII Wajib (4 tipe):

| Tipe PII | Pattern | Contoh | Status |
|---|---|---|---|
| NIK (16 digit) | `\d{16}` with format validation | `3201234506780001` | ✅ |
| Nomor Telepon | `+62xx / 08xx` format | `+6281234567890` | ✅ |
| Alamat Email | Standard email regex | `budi@gmail.com` | ✅ |
| Nomor Rekening (16 digit) | `\d{10,16}` | `1234567890123456` | ✅ |

### PII Bonus (2 tipe tambahan):

| Tipe PII | Pattern | Contoh | Status |
|---|---|---|---|
| NPWP | `XX.XXX.XXX.X-XXX.XXX` | `12.345.678.9-012.345` | ✅ Bonus |
| Nomor Paspor | `[A-Z]{1,2}\d{6,7}` | `AB1234567` | ✅ Bonus |

---

## Constraint 5: Fine-tuning Lokal

**Status**: ✅ COMPLIANT

Model dasar (`indobenchmark/indobert-base-p2`) di-fine-tune menggunakan dataset lokal yang dikurasi secara khusus untuk domain deteksi hoax berbahasa Indonesia.

**Dataset Fine-tuning** (4 sumber CSV, total ~23,711 sampel):

| Sumber | File | Label | Jumlah |
|---|---|---|---|
| TurnBackHoax.id (MAFINDO) | `Cleaned_TurnBackHoax_v3.csv` | `1` (hoax) | ~12,744 |
| Antara News | `Cleaned_Antaranews_v1.csv` | `0` (valid) | ~4,200 |
| Detik.com | `Cleaned_Detik_v2.csv` | `0` (valid) | ~3,267 |
| Kompas.com | `Cleaned_Kompas_v2.csv` | `0` (valid) | ~3,500 |

- **Bahasa**: Indonesia
- **Domain**: Deteksi konten manipulatif / hoax
- **Format CSV**: kolom `url, judul, narasi, label, clean_text`
- **Input model**: Concatenation `judul` + `clean_text` — judul mengandung keyword clickbait yang menjadi sinyal penting untuk deteksi hoax

**Proses Fine-tuning**:
1. Load pre-trained IndoBERT-base-p2
2. Tambahkan classification head (2 kelas: valid, hoax)
3. Gabungkan 4 CSV → split stratified: 10% test, 15% val, 75% train
4. Fine-tune dengan AdamW optimizer + linear warmup schedule
5. Gradient clipping (max_norm=1.0) untuk stabilitas training
6. Evaluasi pada test set terpisah (auto-generated, 2,372 sampel)
7. Simpan model terbaik berdasarkan F1 score ke `models/best_model/`

**Bukti**: Seluruh log training terlihat di `training.ipynb`.
