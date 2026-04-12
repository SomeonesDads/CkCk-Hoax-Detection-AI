# 🛡️ CkCk — Privacy-Aware Hoax Detection AI
> **Track B: The Privacy Brain (NLP / Generative AI)**  
> findIT Hackathon 2026 — Tim CkCk

Sistem Peringatan Dini Konten Manipulatif Berbasis Analisis Teks dan Linguistik untuk Penanganan Krisis Disinformasi AI-Generated di Indonesia.

---

## 🏗️ Architecture

```
Input (Teks Caption / Frame Gambar)
              ↓
        [OCR Engine]           ← jika input berupa gambar/frame video
              ↓                  fallback otomatis ke caption jika
              ↓                  teks hasil OCR tidak memadai
        [PII Filter]           ← sensor NIK, telepon, email, rekening
              ↓                  SELALU berjalan sebelum model apapun
   [IndoBERT Classifier]       ← penentu keputusan utama (100%)
              ↓                  fine-tuned pada dataset hoaks Indonesia
    [Rule-based Detector]      ← berjalan SETELAH classifier
              ↓                  tidak mengubah skor, hanya memperkaya
              ↓                  penjelasan output dengan pola spesifik
      [Output Engine]          ← tentukan 1 dari 4 status +
                                  tulis penjelasan Bahasa Indonesia
```

### Empat Status Output

| Status | Kondisi |
|---|---|
| ✅ TERVERIFIKASI | Confidence valid ≥ 75% |
| ⚠️ KONTEKS BERBEDA | Confidence valid 50–75% |
| 🔍 BELUM TERVERIFIKASI — WASPADAI | Confidence hoaks ≥ 50%, pola manipulatif terdeteksi |
| ❓ BELUM TERVERIFIKASI — NETRAL | Confidence rendah, tidak ada pola manipulatif |

---

## 🧠 Model & Komponen

### 1. IndoBERT Classifier — Penentu Keputusan Utama
- **Base model**: IndoBERT-base-p2 (~110M parameter, jauh di bawah batas 4B)
- **Task**: Sequence classification — hoaks vs valid
- **Fine-tuning**: Dataset MAFINDO/Turnbackhoax.id + IndoNLU benchmark
- **Format**: Disimpan lokal sebagai `.pt` dan ONNX Runtime
- **Runtime**: 100% offline, CPU-compatible via `local_files_only=True`
- Seluruh keputusan status ditentukan oleh confidence score model ini

### 2. Rule-based Detector — Penjelas Hasil
- Berjalan **setelah** classifier, **tidak** ikut menentukan status atau skor
- Mendeteksi pola linguistik spesifik untuk memperkaya penjelasan output
- Menjawab pertanyaan **"kenapa?"**, bukan **"apa hasilnya?"**
- Tiga kategori pola yang dideteksi:
  - **Urgensi palsu**: "SEGERA", "VIRAL", "SEBARKAN", "JANGAN SAMPAI TERHAPUS"
  - **Fear-mongering**: "BAHAYA", "ANCAMAN", "KORBAN", "HANCUR"
  - **Atribusi invalid**: klaim tokoh publik tanpa sumber yang dapat diverifikasi
- Berguna sebagai lapisan transparansi — pengguna tahu **pola spesifik apa** yang mencurigakan

### 3. PII Filter
- Berjalan **pertama** sebelum teks masuk ke komponen manapun
- Berbasis regex, tidak membutuhkan model AI
- **Cakupan wajib**: NIK (16 digit), Nomor telepon (+62xx/08xx), Email, Nomor rekening (16 digit)
- **Cakupan bonus**: Nama lengkap, Alamat, Tanggal lahir
- Teks yang sudah disensor itulah yang diteruskan ke classifier

### 4. OCR Engine
- Preprocessing untuk input berupa gambar atau frame video
- Berjalan lokal menggunakan `pytesseract` / `EasyOCR` — tanpa API eksternal
- Fallback otomatis ke jalur caption jika hasil OCR < 15 karakter

---

## 📁 Project Structure

```
├── src/
│   ├── inferencer.py      # Core inference: load model, prepare input,
│   │                      # jalankan classifier, hitung confidence score
│   ├── pii_filter.py      # PII detection & redaction
│   ├── rule_based.py      # Rule-based pola manipulatif (penjelas hasil)
│   ├── output_engine.py   # Gabung confidence + pola → 4 status +
│   │                      # penjelasan Bahasa Indonesia
│   ├── dataset.py         # Dataset class & data loading
│   ├── preprocessing.py   # Text cleaning & normalisasi
│   ├── trainer.py         # Training loop
│   └── utils.py           # Utilities
├── training.ipynb         # Training notebook (log output terlihat jelas)
├── inference.ipynb        # Clean inference script
├── train_data/            # Dataset training
│   └── labels.csv         # Dipisah SEBELUM preprocessing apapun
├── test_data/             # Dataset test
│   └── labels.csv         # Dipisah SEBELUM preprocessing apapun
├── models/
│   ├── indobert_classifier/      # Model fine-tuned (format HuggingFace)
│   ├── indobert_classifier.onnx  # Model fine-tuned (format ONNX)
│   └── tokenizer/                # Tokenizer disimpan lokal
├── config.yaml            # Path model dan threshold konfigurasi
└── requirements.txt       # Dependencies
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 2. Struktur Data

```
train_data/
└── labels.csv    # kolom: id, text, label (0=valid, 1=hoaks), source, category

test_data/
└── labels.csv    # sama — dipisah SEBELUM preprocessing apapun
```

### 3. Training

Jalankan `training.ipynb` — pastikan seluruh log dan output cell terlihat.

### 4. Inference

Jalankan `inference.ipynb` atau:

```python
from src.inferencer import run_ckck_inference

# Input teks caption
hasil = run_ckck_inference(
    raw_input  = "Sri Mulyani sebut guru itu beban negara, SEBARKAN!",
    input_type = "caption"
)

# Input gambar/frame video
hasil = run_ckck_inference(
    raw_input  = "./frame.jpg",
    input_type = "frame"
)

print(hasil)
# {
#   "status": "TIDAK DITEMUKAN",
#   "confidence_hoax": 0.912,
#   "confidence_valid": 0.088,
#   "penjelasan": "Konten ini memiliki karakteristik kuat sebagai konten
#                  manipulatif. Ditemukan pola linguistik yang perlu
#                  diwaspadai: urgensi_palsu ('SEBARKAN').",
#   "pola_terdeteksi": ["urgensi_palsu"],
#   "pii_disensor": false
# }
```

---

## ⚙️ Constraint Compliance

| Constraint | Detail | Status |
|---|---|---|
| Model ≤ 4B parameter | IndoBERT-base-p2 ~110M parameter | ✅ |
| Offline Total | `local_files_only=True`, ONNX `CPUExecutionProvider`, zero network call | ✅ |
| PII Filter di dalam pipeline | Berjalan di `inference.ipynb` sebelum model apapun | ✅ |
| Cakupan PII | NIK, Telepon, Email, Rekening + bonus types | ✅ |
| Fine-tuning Lokal | Fine-tuned pada dataset MAFINDO lokal, bukan pretrained mentah | ✅ |

---

## 👥 Tim CkCk

findIT Hackathon 2026
