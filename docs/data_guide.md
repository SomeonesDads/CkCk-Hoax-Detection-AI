# Data Collection & Preparation Guide

## Overview

This guide covers how to collect, clean, and prepare training data for the hoax detection model.

## Current Dataset

The training data consists of **4 cleaned CSV files** in `train_data/`:

| File | Source | Label | Description |
|---|---|---|---|
| `Cleaned_TurnBackHoax_v3.csv` | TurnBackHoax.id (MAFINDO) | `1` (hoax) | Verified hoax articles |
| `Cleaned_Antaranews_v1.csv` | Antara News | `0` (valid) | Legitimate news |
| `Cleaned_Detik_v2.csv` | Detik.com | `0` (valid) | Legitimate news |
| `Cleaned_Kompas_v2.csv` | Kompas.com | `0` (valid) | Legitimate news |

### CSV Schema

```csv
url,judul,narasi,label,clean_text
https://example.com/article,Judul Artikel,Isi narasi lengkap...,0,isi narasi lengkap yang sudah dibersihkan...
```

| Column | Type | Description |
|---|---|---|
| `url` | string | Source URL of the article |
| `judul` | string | Article title/headline |
| `narasi` | string | Raw article body text |
| `label` | int | `0` = valid, `1` = hoax |
| `clean_text` | string | Pre-cleaned, lowercased version of `narasi` |

### Training Input Format

For training, the model receives a concatenation of **title + body**:
```
{judul}. {clean_text}
```
This leverages clickbait-style titles that often contain keywords used to mislead readers.

## Data Pipeline

The `create_dataloaders()` function in `src/dataset.py` handles:

1. **Loading**: Reads all 4 CSVs and concatenates them
2. **Concatenation**: Combines `judul` + `clean_text` as model input
3. **Test split**: Holds out 10% stratified for testing → saved to `test_data/test.csv`
4. **Train/val split**: Splits remaining 90% into 85% train / 15% validation
5. **Tokenization**: IndoBERT WordPiece tokenization (max 256 tokens)

## Data Sources

### 1. TurnBackHoax.id (MAFINDO)

The primary Indonesian fact-checking database maintained by MAFINDO (Masyarakat Anti Fitnah Indonesia).

- **URL**: https://turnbackhoax.id
- **Content**: Verified hoax articles with labels
- **Labels available**: Hoax, Misleading, Fabricated, Satire
- **Estimated size**: 5,000+ entries
- **How to get**: Web scraping or manual collection

### 2. CekFakta.com

Collaborative fact-checking platform by Indonesian media organizations.

- **URL**: https://cekfakta.com
- **Content**: Fact-checked claims with verdicts
- **How to get**: Web scraping

### 3. Legitimate News Sources

For "valid" (non-hoax) samples, collected from reputable sources:

- `antaranews.com` — National news agency
- `kompas.com` — National news
- `detik.com` — National news
- `tempo.co` — Investigative journalism
- `tirto.id` — In-depth reporting
- `bbc.com/indonesia` — International perspective

### 4. Public Datasets

- **Indonesian Hoax News Dataset** (Kaggle)
  - Search: "indonesian hoax news" on kaggle.com
- **IndoNLU Benchmark** (GitHub: indobenchmark)
  - Various Indonesian NLP datasets
- **Nusa Menulis** corpus
  - Indonesian writing samples

### 5. Social Media / Chain Messages

For realistic hoax samples:
- WhatsApp forwarded messages (anonymized)
- Facebook posts flagged as misinformation
- Twitter/X threads with false claims

## Data Collection Script

A basic scraping template (uncomment `beautifulsoup4` and `requests` in requirements.txt):

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_news(url, label, source):
    """Scrape a single news article."""
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Adjust selectors per site
        title = soup.find('h1').get_text(strip=True)
        body = soup.find('article').get_text(strip=True)
        
        return {
            'text': f"{title}. {body}",
            'label': label,
            'source': source,
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Example usage:
# article = scrape_news("https://kompas.com/article/123", label=0, source="kompas.com")
```

## Data Preparation Checklist

- [x] Collect minimum **1,000 hoax** samples
- [x] Collect minimum **1,000 valid** samples
- [x] Ensure balanced class distribution (aim for 50/50)
- [x] Remove duplicates
- [x] Remove very short texts (< 20 characters)
- [x] Remove texts that are purely URLs or images
- [ ] Anonymize any personally identifiable information
- [x] Auto-split into train/val/test via `config.yaml` settings
- [x] Validate CSV format loads correctly

## Data Augmentation (if limited data)

If you have fewer than 1,000 samples per class, consider:

1. **Back-translation**: Translate ID → EN → ID to create paraphrases
2. **Synonym replacement**: Replace words with Indonesian synonyms
3. **Random insertion/deletion**: Slight text modifications
4. **Cross-domain transfer**: Use hoax patterns from other languages

## Target Metrics

| Metric | Minimum Target | Ideal |
|---|---|---|
| Accuracy | > 80% | > 90% |
| F1 Score | > 0.78 | > 0.88 |
| Precision | > 0.75 | > 0.85 |
| Recall | > 0.80 | > 0.90 |
