"""
Dataset & Data Loading
========================
Custom PyTorch Dataset for Indonesian hoax detection.
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
    
    Expects a CSV file with columns: text, label
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


def load_data_from_csv(csv_path: str, text_column: str = "text", label_column: str = "label") -> tuple:
    """
    Load text and labels from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        text_column: Name of the text column.
        label_column: Name of the label column.
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in {csv_path}. Available: {list(df.columns)}")

    # Drop rows with missing values
    df = df.dropna(subset=[text_column, label_column])

    texts = df[text_column].tolist()
    labels = df[label_column].astype(int).tolist()

    print(f"[INFO] Loaded {len(texts)} samples from {csv_path}")
    print(f"[INFO] Label distribution: {df[label_column].value_counts().to_dict()}")

    return texts, labels


def create_dataloaders(config_path: str = "config.yaml") -> tuple:
    """
    Create train, validation, and test DataLoaders from config.
    
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

    # Load training data
    train_texts, train_labels = load_data_from_csv(
        data_cfg["train_path"],
        data_cfg["text_column"],
        data_cfg["label_column"],
    )

    # Limit samples if configured
    max_samples = data_cfg.get("max_samples")
    if max_samples and max_samples < len(train_texts):
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        print(f"[INFO] Limited to {max_samples} training samples")

    # Split train/val
    val_split = data_cfg.get("val_split", 0.15)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=val_split,
        random_state=train_cfg.get("seed", 42),
        stratify=train_labels,
    )

    print(f"[INFO] Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Create datasets
    train_dataset = HoaxDataset(train_texts, train_labels, tokenizer, model_cfg["max_length"])
    val_dataset = HoaxDataset(val_texts, val_labels, tokenizer, model_cfg["max_length"])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    # Load test data if exists
    test_loader = None
    test_path = data_cfg.get("test_path")
    if test_path and os.path.exists(test_path):
        test_texts, test_labels = load_data_from_csv(
            test_path, data_cfg["text_column"], data_cfg["label_column"]
        )
        test_dataset = HoaxDataset(test_texts, test_labels, tokenizer, model_cfg["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
        print(f"[INFO] Test: {len(test_texts)}")

    return train_loader, val_loader, test_loader, tokenizer
