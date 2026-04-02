# Data Collection & Preparation Guide

## Overview

This guide covers how to collect, clean, and prepare training data for the hoax detection model.

## Required Format

### `labels.csv` Schema

```csv
id,text,label,source,category
article_001,"Full article text here...",0,kompas.com,news
article_002,"Hoax article text here...",1,whatsapp,chain_message
```

| Column | Type | Values |
|---|---|---|
| `id` | string | Unique identifier |
| `text` | string | Full article/message text |
| `label` | int | `0` = valid, `1` = hoax |
| `source` | string | Origin (kompas.com, whatsapp, etc.) |
| `category` | string | `news`, `social_media`, `chain_message` |

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

For "valid" (non-hoax) samples, collect from reputable sources:

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

- [ ] Collect minimum **1,000 hoax** samples
- [ ] Collect minimum **1,000 valid** samples
- [ ] Ensure balanced class distribution (aim for 50/50)
- [ ] Remove duplicates
- [ ] Remove very short texts (< 20 characters)
- [ ] Remove texts that are purely URLs or images
- [ ] Anonymize any personally identifiable information
- [ ] Split into `train_data/labels.csv` and `test_data/labels.csv` (80/20)
- [ ] Validate CSV format loads correctly

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
