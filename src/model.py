"""
Model Definition & Inference
==============================
Fine-tuned IndoBERT-base-p2 for Indonesian hoax classification.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import os


class HoaxDetector:
    """
    Hoax detection model wrapping a fine-tuned IndoBERT classifier.
    
    Usage:
        detector = HoaxDetector.from_config("config.yaml")
        result = detector.predict("Berita ini sangat mencurigakan...")
    """

    LABEL_MAP = {0: "VALID", 1: "HOAX"}

    def __init__(self, model_name: str, num_labels: int = 2, max_length: int = 256, device: str = "cpu"):
        self.device = torch.device(device)
        self.max_length = max_length
        self.num_labels = num_labels
        self.model_name = model_name

        self.tokenizer = None
        self.model = None

    def load_pretrained(self):
        """Load the pre-trained IndoBERT model and tokenizer."""
        print(f"[INFO] Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model loaded on {self.device}")
        return self

    def load_finetuned(self, model_path: str):
        """Load a fine-tuned model from a saved checkpoint."""
        print(f"[INFO] Loading fine-tuned model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Fine-tuned model loaded on {self.device}")
        return self

    def predict(self, text: str) -> dict:
        """
        Predict whether a text is hoax or valid.
        
        Args:
            text: Input text to classify.
            
        Returns:
            dict with keys: label, confidence, probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() or load_finetuned() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        return {
            "label": self.LABEL_MAP[pred_idx],
            "label_id": pred_idx,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.LABEL_MAP[i]: round(probs[0][i].item(), 4)
                for i in range(self.num_labels)
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict on a batch of texts."""
        return [self.predict(text) for text in texts]

    def save(self, save_path: str):
        """Save the fine-tuned model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[INFO] Model saved to {save_path}")

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "HoaxDetector":
        """Create a HoaxDetector from a config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        inference_cfg = config.get("inference", {})

        detector = cls(
            model_name=model_cfg["name"],
            num_labels=model_cfg["num_labels"],
            max_length=model_cfg["max_length"],
            device=inference_cfg.get("device", "cpu"),
        )
        return detector


if __name__ == "__main__":
    # Quick test
    detector = HoaxDetector.from_config("config.yaml")
    detector.load_pretrained()
    result = detector.predict("Ini adalah berita uji coba untuk deteksi hoax.")
    print(result)
