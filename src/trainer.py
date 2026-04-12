"""
Training Loop
===============
Fine-tuning pipeline for IndoBERT hoax classifier.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import yaml
import os
import time
import json
from datetime import datetime

from .dataset import create_dataloaders


class Trainer:
    """
    Training pipeline for the hoax detection model.
    
    Usage:
        trainer = Trainer.from_config("config.yaml")
        trainer.train()
        trainer.evaluate()
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.paths_cfg = config["paths"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.tokenizer = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

    def setup(self):
        """Initialize model, data, optimizer, and scheduler."""
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Setting up training pipeline...")

        # Load data
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = \
            create_dataloaders("config.yaml")

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_cfg["name"],
            num_labels=self.model_cfg["num_labels"],
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.train_cfg["epochs"]
        warmup_steps = int(total_steps * self.train_cfg["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"[INFO] Model: {self.model_cfg['name']}")
        print(f"[INFO] Total steps: {total_steps}, Warmup: {warmup_steps}")

        return self

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation. Returns metrics dict."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

    def train(self):
        """Full training loop."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

        best_f1 = 0
        epochs = self.train_cfg["epochs"]

        for epoch in range(epochs):
            start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            elapsed = time.time() - start

            # Log
            print(f"\nEpoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f}")

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Save best model
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_path = os.path.join(self.paths_cfg["model_dir"], "best_model")
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"  ✓ New best model saved (F1: {best_f1:.4f})")

        print("\n" + "=" * 60)
        print(f"Training complete! Best Val F1: {best_f1:.4f}")
        print("=" * 60)

        # Save training history
        history_path = os.path.join(self.paths_cfg["model_dir"], "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set and print classification report."""
        if self.test_loader is None:
            print("[WARN] No test data available.")
            return

        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in self.test_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        label_names = ["VALID", "HOAX"]
        report = classification_report(all_labels, all_preds, target_names=label_names)
        print("\n📊 Test Set Classification Report:")
        print(report)

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
        }

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "Trainer":
        """Create Trainer from config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)


if __name__ == "__main__":
    trainer = Trainer.from_config("config.yaml")
    trainer.setup()
    trainer.train()
    trainer.evaluate()
