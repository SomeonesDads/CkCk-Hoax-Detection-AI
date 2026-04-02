# System Architecture

## Overview

```
┌──────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                      │
│                                                            │
│  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐ │
│  │  Input   │──▶│PII Filter │──▶│Preprocess│──▶│IndoBERT│ │
│  │  Text    │   │ (Redact)  │   │ (Clean)  │   │Classify│ │
│  └─────────┘   └───────────┘   └──────────┘   └────────┘ │
│                      │                              │      │
│                      ▼                              ▼      │
│               Redacted Text              Prediction        │
│               + PII Report          (HOAX / VALID)         │
│                                    + Confidence Score      │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. PII Filter (`src/pii_filter.py`)

**Purpose**: Detect and redact Personally Identifiable Information before processing.

**Supported PII Types**:
| Type | Pattern | Example |
|---|---|---|
| NIK | 16-digit national ID | `3201234506780001` |
| Phone | +62xx / 08xx | `+6281234567890` |
| Email | standard format | `user@email.com` |
| Bank Account | 10-16 digits | `1234567890123456` |
| NPWP | Tax ID format | `12.345.678.9-012.345` |
| Passport | 1-2 letters + 6-7 digits | `AB1234567` |

**Design Decision**: Regex-based approach chosen over NER-based because:
- Zero model overhead (no extra parameters)
- Deterministic results (no false negatives on known patterns)
- Instant execution on CPU
- Indonesian PII formats are highly structured

### 2. Text Preprocessor (`src/preprocessing.py`)

**Purpose**: Clean and normalize Indonesian text for model input.

**Pipeline**:
1. Remove HTML tags
2. Remove URLs
3. Remove @mentions and #hashtags
4. Remove emojis
5. Lowercase
6. Reduce excessive punctuation
7. (Optional) Indonesian slang normalization
8. (Optional) Sastrawi stemming
9. Normalize whitespace

### 3. Hoax Classifier (`src/model.py`)

**Purpose**: Fine-tuned IndoBERT for binary classification (hoax vs valid).

**Model**: `indobenchmark/indobert-base-p2`
- Architecture: BERT-base (12 layers, 768 hidden, 12 heads)
- Parameters: ~110M (well within ≤ 4B constraint)
- Pre-training: Indonesian Wikipedia + news corpus
- Fine-tuning: Sequence classification head

### 4. Training Pipeline (`src/trainer.py`)

**Features**:
- AdamW optimizer with linear warmup schedule
- Gradient clipping (max_norm=1.0)
- Best model checkpointing (by val F1)
- Training history logging (JSON)
- Full classification report on test set

## Data Flow

```
[Raw Text]
    │
    ▼
[PII Filter] ─── detects NIK, phone, email, bank account
    │              masks with █ characters
    ▼              returns redacted text + PII report
[Preprocessor] ── lowercases, removes noise
    │              normalizes Indonesian text
    ▼
[Tokenizer] ──── IndoBERT WordPiece tokenization
    │             max_length=256 tokens
    ▼
[IndoBERT] ────── forward pass through fine-tuned model
    │              softmax over [VALID, HOAX]
    ▼
[Output] ──────── label, confidence, probabilities
```

## Directory Structure

```
CkCk-Hoax-Detection-AI/
├── src/                  # Source modules
│   ├── model.py          # HoaxDetector class
│   ├── dataset.py        # HoaxDataset + DataLoader creation
│   ├── pii_filter.py     # PIIFilter class
│   ├── preprocessing.py  # TextPreprocessor class
│   ├── trainer.py        # Training pipeline
│   └── utils.py          # Shared utilities
├── docs/                 # Documentation
├── train_data/           # Training CSV + raw texts
├── test_data/            # Test CSV + raw texts
├── models/               # Saved model checkpoints
├── training.ipynb        # Training notebook (with logs)
├── inference.ipynb       # Clean inference notebook
├── config.yaml           # All configuration
└── requirements.txt      # Python dependencies
```
