"""
Dataset & Data Loading
========================
Custom PyTorch Dataset for Indonesian hoax detection.
Supports loading multiple CSV files with judul+clean_text concatenation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import yaml
import os


class HoaxDataset(Dataset):
    """
    PyTorch Dataset for hoax detection.
    
    Expects a list of texts and labels.
    Label: 0 = valid, 1 = hoax
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data_from_csv(csv_path: str, text_column: str = "text", label_column: str = "label",
                       title_column: str = None) -> tuple:
    """
    Load text and labels from a single CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        text_column: Name of the text column.
        label_column: Name of the label column.
        title_column: Optional. If provided, concatenates title + text.
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in {csv_path}. Available: {list(df.columns)}")

    # Drop rows with missing values in key columns
    required_cols = [text_column, label_column]
    if title_column and title_column in df.columns:
        required_cols.append(title_column)
    df = df.dropna(subset=required_cols)

    # Concatenate title + text if title_column is provided
    if title_column and title_column in df.columns:
        df["_combined_text"] = df[title_column].astype(str) + ". " + df[text_column].astype(str)
        texts = df["_combined_text"].tolist()
    else:
        texts = df[text_column].tolist()

    labels = df[label_column].astype(int).tolist()

    print(f"[INFO] Loaded {len(texts)} samples from {os.path.basename(csv_path)}")
    print(f"[INFO] Label distribution: {df[label_column].value_counts().to_dict()}")

    return texts, labels


def load_multi_csv(data_dir: str, file_list: list[str], text_column: str = "clean_text",
                   label_column: str = "label", title_column: str = None) -> tuple:
    """
    Load and merge multiple CSV files from a directory.
    
    Args:
        data_dir: Directory containing CSV files.
        file_list: List of CSV filenames to load.
        text_column: Name of the text column.
        label_column: Name of the label column.
        title_column: Optional. If provided, concatenates title + text.
        
    Returns:
        Tuple of (texts, labels, source_names)
    """
    all_texts = []
    all_labels = []

    print(f"[INFO] Loading {len(file_list)} CSV files from {data_dir}/")
    print("=" * 50)

    for filename in file_list:
        csv_path = os.path.join(data_dir, filename)
        if not os.path.exists(csv_path):
            print(f"[WARN] File not found: {csv_path}, skipping.")
            continue
        texts, labels = load_data_from_csv(csv_path, text_column, label_column, title_column)
        all_texts.extend(texts)
        all_labels.extend(labels)

    print("=" * 50)
    total_valid = sum(1 for l in all_labels if l == 0)
    total_hoax = sum(1 for l in all_labels if l == 1)
    print(f"[INFO] Total combined: {len(all_texts)} samples")
    print(f"[INFO] Combined distribution: {{0 (valid): {total_valid}, 1 (hoax): {total_hoax}}}")

    return all_texts, all_labels


def create_dataloaders(config_path: str = "config.yaml") -> tuple:
    """
    Create train, validation, and test DataLoaders from config.
    
    Supports two modes:
    - Multi-file: config has 'train_dir' + 'train_files' (new format)
    - Single-file: config has 'train_path' (legacy format)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    title_column = data_cfg.get("title_column", None)

    # ── Load training data ──────────────────────────────────────────
    if "train_files" in data_cfg and "train_dir" in data_cfg:
        # New multi-file mode
        all_texts, all_labels = load_multi_csv(
            data_dir=data_cfg["train_dir"],
            file_list=data_cfg["train_files"],
            text_column=data_cfg["text_column"],
            label_column=data_cfg["label_column"],
            title_column=title_column,
        )
    elif "train_path" in data_cfg:
        # Legacy single-file mode
        all_texts, all_labels = load_data_from_csv(
            data_cfg["train_path"],
            data_cfg["text_column"],
            data_cfg["label_column"],
            title_column=title_column,
        )
    else:
        raise ValueError("Config must specify either 'train_dir'+'train_files' or 'train_path'.")

    # Limit samples if configured
    max_samples = data_cfg.get("max_samples")
    if max_samples and max_samples < len(all_texts):
        all_texts = all_texts[:max_samples]
        all_labels = all_labels[:max_samples]
        print(f"[INFO] Limited to {max_samples} samples")

    seed = train_cfg.get("seed", 42)

    # ── Split off test set ──────────────────────────────────────────
    test_split = data_cfg.get("test_split", 0.0)
    test_loader = None

    if test_split > 0:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels,
            test_size=test_split,
            random_state=seed,
            stratify=all_labels,
        )
        # Save test set for reproducibility
        test_path = data_cfg.get("test_path", "test_data/test.csv")
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
        test_df.to_csv(test_path, index=False)
        print(f"[INFO] Test set saved to {test_path} ({len(test_texts)} samples)")
    else:
        train_texts = all_texts
        train_labels = all_labels
        test_texts = None
        test_labels = None

    # ── Split train into train + val ────────────────────────────────
    val_split = data_cfg.get("val_split", 0.15)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=val_split,
        random_state=seed,
        stratify=train_labels,
    )

    print(f"\n[INFO] Final splits:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val:   {len(val_texts)}")
    if test_texts:
        print(f"  Test:  {len(test_texts)}")

    # ── Create datasets ─────────────────────────────────────────────
    train_dataset = HoaxDataset(train_texts, train_labels, tokenizer, model_cfg["max_length"])
    val_dataset = HoaxDataset(val_texts, val_labels, tokenizer, model_cfg["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    if test_texts:
        test_dataset = HoaxDataset(test_texts, test_labels, tokenizer, model_cfg["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    # ── Load existing test file (legacy fallback) ───────────────────
    if test_loader is None:
        test_path = data_cfg.get("test_path")
        if test_path and os.path.exists(test_path):
            test_texts_f, test_labels_f = load_data_from_csv(
                test_path, "text", data_cfg["label_column"]
            )
            test_dataset = HoaxDataset(test_texts_f, test_labels_f, tokenizer, model_cfg["max_length"])
            test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
            print(f"[INFO] Test: {len(test_texts_f)} (loaded from {test_path})")

    return train_loader, val_loader, test_loader, tokenizer
