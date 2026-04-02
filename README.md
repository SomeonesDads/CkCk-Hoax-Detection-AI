# 🛡️ CkCk — Privacy-Aware Hoax Detection AI

> **Track B: The Privacy Brain (NLP / Generative AI)**  
> findIT Hackathon 2026 — Tim CkCk

Sistem Peringatan Dini Konten Manipulatif Berbasis Analisis Teks dan Linguistik untuk Penanganan Krisis Disinformasi AI-Generated di Indonesia.

## 🏗️ Architecture

```
Input Text → [PII Filter] → [Preprocessor] → [IndoBERT Classifier] → Prediction (Hoax/Valid)
                  ↓
           Redacted Output
```

- **Model**: IndoBERT-base-p2 (~110M params, fine-tuned)
- **PII Filter**: Regex + rule-based pipeline (NIK, phone, email, bank account)
- **Inference**: 100% offline, CPU-compatible

## 📁 Project Structure

```
├── src/                     # Source code modules
│   ├── model.py             # Model definition & loading
│   ├── dataset.py           # Dataset class & data loading
│   ├── pii_filter.py        # PII detection & redaction
│   ├── preprocessing.py     # Text cleaning
│   ├── trainer.py           # Training loop
│   └── utils.py             # Utilities
├── training.ipynb           # Training notebook (with logs)
├── inference.ipynb          # Clean inference script
├── train_data/              # Training dataset
├── test_data/               # Test dataset
├── models/                  # Saved model weights
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your training data in `train_data/` and test data in `test_data/` following this structure:

```
train_data/
├── labels.csv    # Columns: id, text, label (0=valid, 1=hoax), source, category
```

### 3. Train

Run `training.ipynb` or:

```bash
python -m src.trainer
```

### 4. Inference

Run `inference.ipynb` or:

```bash
python -m src.model --input "Your text here"
```

## ⚙️ Constraint Compliance

| Constraint | Status |
|---|---|
| Model ≤ 4B params | ✅ IndoBERT ~110M |
| Offline Total | ✅ No API calls |
| PII Filter | ✅ Integrated pipeline |
| PII Coverage (NIK, Phone, Email, Bank) | ✅ + Bonus types |
| Fine-tuning Lokal | ✅ Domain-specific dataset |

## 👥 Tim CkCk

findIT Hackathon 2026