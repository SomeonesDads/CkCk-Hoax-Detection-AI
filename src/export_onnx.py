"""
ONNX Export Utility
=====================
Exports a fine-tuned IndoBERT model to ONNX format for CPU-optimized inference.

Usage:
    python src/export_onnx.py models/best_model_v2
"""

import torch
import os
import sys

# Fix Windows Unicode encoding issue
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_to_onnx(
    model_path: str,
    output_path: str = None,
    max_length: int = 256,
    num_labels: int = 2,
    opset_version: int = 14,
):
    """
    Export a saved HuggingFace model to ONNX format.

    Args:
        model_path: Path to the saved model directory (e.g., 'models/best_model_v2')
        output_path: Path for the ONNX file. Defaults to '{model_path}/model.onnx'
        max_length: Max sequence length for the model
        num_labels: Number of output labels
        opset_version: ONNX opset version
    """
    if output_path is None:
        output_path = os.path.join(model_path, "model.onnx")

    print(f"[INFO] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    # Create dummy input
    dummy_text = "Contoh teks untuk ekspor ONNX."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    print(f"[INFO] Exporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for compatibility
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] ✅ ONNX model saved to: {output_path}")
    print(f"[INFO]    Size: {file_size_mb:.1f} MB")

    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
        ort_outputs = session.run(None, ort_inputs)
        print(f"[INFO] ✅ ONNX Runtime verification passed")
        print(f"[INFO]    Output shape: {ort_outputs[0].shape}")
    except ImportError:
        print("[INFO] onnxruntime not installed — skipping verification")
        print("[INFO] Install with: pip install onnxruntime")

    return output_path


if __name__ == "__main__":
    model_path = "models/best_model_v3"
    export_to_onnx(model_path)