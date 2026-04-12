"""
Model Definition & Inference
==============================
Fine-tuned IndoBERT-base-p2 for Indonesian hoax classification.
Includes Output Engine with 4-status system and Rule-based Detector integration.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import yaml
import os
import numpy as np
import onnxruntime as ort

from .rule_detector import RuleBasedDetector


# ── 4-Status Output Definitions ───────────────────────────────────────

STATUS_TERVERIFIKASI = "TERVERIFIKASI"
STATUS_KONTEKS_BERBEDA = "KONTEKS BERBEDA"
STATUS_BELUM_WASPADAI = "BELUM TERVERIFIKASI — WASPADAI"
STATUS_BELUM_NETRAL = "BELUM TERVERIFIKASI — NETRAL"

STATUS_DESCRIPTIONS = {
    STATUS_TERVERIFIKASI: "Konten kemungkinan besar akurat dan dapat dipercaya",
    STATUS_KONTEKS_BERBEDA: "Konten mungkin benar namun perlu konfirmasi konteks tambahan",
    STATUS_BELUM_WASPADAI: "Indikasi kuat sebagai konten manipulatif dengan pola spesifik",
    STATUS_BELUM_NETRAL: "Tidak cukup bukti untuk klasifikasi definitif",
}


class HoaxDetector:
    """
    Hoax detection model wrapping a fine-tuned IndoBERT classifier.
    
    Includes:
    - IndoBERT sequence classifier (primary decision maker)
    - Rule-based Detector (explainability layer)
    - Output Engine (4-status mapping)
    
    Usage:
        detector = HoaxDetector.from_config("config.yaml")
        result = detector.predict("Berita ini sangat mencurigakan...")
        print(result["status"])       # e.g., "BELUM TERVERIFIKASI — WASPADAI"
        print(result["explanation"])  # Human-readable Bahasa Indonesia explanation
    """

    LABEL_MAP = {0: "VALID", 1: "HOAX"}

    def __init__(self, model_name: str, num_labels: int = 2, max_length: int = 256, device: str = "cpu"):
        self.device = torch.device(device)
        self.max_length = max_length
        self.num_labels = num_labels
        self.model_name = model_name

        self.tokenizer = None
        self.model = None

        # ONNX support
        self.onnx_mode = False
        self.ort_session = None

        # Rule-based detector for explainability
        self.rule_detector = RuleBasedDetector()

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
        """Load a fine-tuned model from a saved local checkpoint."""
        print(f"[INFO] Loading fine-tuned model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.onnx_mode = False
        print(f"[INFO] Fine-tuned model loaded on {self.device} (offline)")
        return self

    def load_onnx(self, model_path: str):
        """
        Load a fine-tuned model in ONNX format for optimized CPU inference.
        
        Args:
            model_path: Path to model directory or .onnx file.
        """
        if os.path.isdir(model_path):
            onnx_path = os.path.join(model_path, "model.onnx")
        else:
            onnx_path = model_path

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        print(f"[INFO] Loading ONNX model from: {onnx_path}")
        
        # Load tokenizer from the same directory if possible
        tokenizer_dir = os.path.dirname(onnx_path) if os.path.isfile(model_path) else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        # Initialize ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.onnx_mode = True
        
        print(f"[INFO] ONNX model loaded successfully (CPU optimized)")
        return self

    def _classify(self, text: str) -> dict:
        """
        Raw classification — returns label, confidence, and probabilities.
        
        Args:
            text: Input text to classify.
            
        Returns:
            dict with keys: label, label_id, confidence, probabilities
        """
        if self.tokenizer is None or (self.model is None and self.ort_session is None):
            raise RuntimeError("Model not loaded. Call load_pretrained(), load_finetuned(), or load_onnx() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.onnx_mode:
            # ONNX Inference Path
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            # Convert to probabilities using numpy-based softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            pred_idx = np.argmax(probs, axis=-1)[0]
            confidence = probs[0][pred_idx]
            
            return {
                "label": self.LABEL_MAP[pred_idx],
                "label_id": int(pred_idx),
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    self.LABEL_MAP[i]: round(float(probs[0][i]), 4)
                    for i in range(self.num_labels)
                },
            }
        else:
            # PyTorch Inference Path
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

    def _determine_status(self, classification: dict, rule_summary: dict) -> dict:
        """
        Output Engine: Map classifier confidence + rule patterns to 4-status system.
        
        Status Logic:
            - TERVERIFIKASI:             valid confidence ≥ 75%
            - KONTEKS BERBEDA:           valid confidence 50–75%
            - BELUM TERVERIFIKASI — WASPADAI: hoax confidence ≥ 50% AND manipulative patterns found
            - BELUM TERVERIFIKASI — NETRAL:   low confidence OR no manipulative patterns
        
        Args:
            classification: Output from _classify()
            rule_summary: Output from rule_detector.get_summary()
            
        Returns:
            dict with status, description, and explanation
        """
        valid_conf = classification["probabilities"]["VALID"]
        hoax_conf = classification["probabilities"]["HOAX"]
        has_patterns = rule_summary["has_manipulative_patterns"]

        # Determine status
        if valid_conf >= 0.75:
            status = STATUS_TERVERIFIKASI
        elif valid_conf >= 0.50:
            status = STATUS_KONTEKS_BERBEDA
        elif hoax_conf >= 0.50 and has_patterns:
            status = STATUS_BELUM_WASPADAI
        else:
            status = STATUS_BELUM_NETRAL

        # Build explanation in Bahasa Indonesia
        explanation_parts = []

        if status == STATUS_TERVERIFIKASI:
            explanation_parts.append(
                f"Model mendeteksi konten ini sebagai valid dengan tingkat kepercayaan {valid_conf*100:.1f}%."
            )
        elif status == STATUS_KONTEKS_BERBEDA:
            explanation_parts.append(
                f"Model memberikan kepercayaan {valid_conf*100:.1f}% bahwa konten ini valid, "
                "namun belum cukup tinggi untuk verifikasi penuh. Disarankan untuk mengecek konteks tambahan."
            )
        elif status == STATUS_BELUM_WASPADAI:
            explanation_parts.append(
                f"Model mendeteksi kemungkinan hoaks dengan kepercayaan {hoax_conf*100:.1f}%."
            )
            # Add pattern explanations
            for cat, details in rule_summary["details"].items():
                patterns_found = [d["description"] for d in details[:3]]  # Limit to 3 per category
                explanation_parts.append(
                    f"Pola {cat} terdeteksi: " + "; ".join(patterns_found) + "."
                )
        else:  # NETRAL
            explanation_parts.append(
                "Tidak cukup bukti untuk mengklasifikasikan konten ini secara definitif. "
                "Disarankan verifikasi manual melalui sumber terpercaya."
            )

        return {
            "status": status,
            "status_description": STATUS_DESCRIPTIONS[status],
            "explanation": " ".join(explanation_parts),
        }

    def predict(self, text: str) -> dict:
        """
        Full prediction pipeline with 4-status output.
        
        Pipeline: Text → Classifier → Rule Detector → Output Engine → Result
        
        Args:
            text: Input text to classify.
            
        Returns:
            dict with:
                - status: One of 4 statuses
                - status_description: Short description
                - explanation: Detailed Bahasa Indonesia explanation
                - label: Raw classifier label (VALID/HOAX)
                - confidence: Raw classifier confidence
                - probabilities: Per-class probabilities
                - patterns: Detected manipulative patterns
        """
        # Step 1: Classify with IndoBERT
        classification = self._classify(text)

        # Step 2: Detect manipulative patterns
        rule_summary = self.rule_detector.get_summary(text)

        # Step 3: Determine 4-status output
        output = self._determine_status(classification, rule_summary)

        # Combine all results
        return {
            **output,
            "label": classification["label"],
            "label_id": classification["label_id"],
            "confidence": classification["confidence"],
            "probabilities": classification["probabilities"],
            "patterns": rule_summary,
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
        # Store use_onnx preference
        detector.use_onnx = inference_cfg.get("use_onnx", False)
        return detector


if __name__ == "__main__":
    # Quick test
    detector = HoaxDetector.from_config("config.yaml")
    detector.load_pretrained()

    test_texts = [
        "Pemerintah Indonesia mengumumkan kebijakan ekonomi baru untuk investasi.",
        "SEGERA SEBARKAN!! Vaksin COVID terbukti mengandung RACUN!! BAHAYA!! Menurut ahli kesehatan ini VIRAL!!",
    ]

    for text in test_texts:
        result = detector.predict(text)
        print(f"Text:   {text[:80]}...")
        print(f"Status: {result['status']}")
        print(f"Expl:   {result['explanation']}")
        print(f"Label:  {result['label']} ({result['confidence']*100:.1f}%)")
        print()
