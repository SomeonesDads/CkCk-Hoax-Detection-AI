import os
import sys
import argparse
import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export IndoBERT ke ONNX + simpan tokenizer lokal")
    parser.add_argument("--config",     default="config.yaml", help="Path ke config.yaml")
    parser.add_argument("--verify",     action="store_true",   help="Verifikasi output ONNX setelah export")
    parser.add_argument("--simplify",   action="store_true",   help="Jalankan onnxsim untuk optimasi graph (perlu onnxsim)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_tokenizer(config: dict) -> str:
    """Simpan tokenizer dari best_model ke direktori lokal terpisah."""
    best_model_path  = os.path.join(config["paths"]["model_dir"], "best_model")
    tokenizer_path   = config["inference"].get("tokenizer_path", "models/tokenizer")

    if not os.path.exists(best_model_path):
        print(f"❌ best_model tidak ditemukan di: {best_model_path}")
        print("   Jalankan training.ipynb terlebih dahulu.")
        sys.exit(1)

    print(f"[1/3] Menyimpan tokenizer dari: {best_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)

    saved_files = os.listdir(tokenizer_path)
    print(f"      ✅ Tokenizer disimpan ke: {tokenizer_path}")
    print(f"      Files: {saved_files}")
    return tokenizer_path


def export_onnx(config: dict, verify: bool = True, simplify: bool = False) -> str:
    """Export model PyTorch ke format ONNX."""
    best_model_path = os.path.join(config["paths"]["model_dir"], "best_model")
    onnx_output     = config["inference"]["model_path"]
    max_length      = config["model"]["max_length"]
    num_labels      = config["model"]["num_labels"]

    os.makedirs(os.path.dirname(onnx_output) or ".", exist_ok=True)

    # ── Load PyTorch model ────────────────────────────────────────────────────
    print(f"\n[2/3] Loading PyTorch model dari: {best_model_path}")
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path,
        num_labels=num_labels,
    )
    pt_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    # ── Buat dummy input ──────────────────────────────────────────────────────
    dummy_text  = "contoh kalimat untuk tracing ONNX graph model IndoBERT"
    dummy_input = tokenizer(
        dummy_text,
        max_length     = max_length,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    print(f"      Dummy input shape: input_ids={dummy_input['input_ids'].shape}")

    # ── Export ke ONNX ────────────────────────────────────────────────────────
    print(f"      Mengekspor ke: {onnx_output} (opset 14)...")
    with torch.no_grad():
        torch.onnx.export(
            pt_model,
            args           = (dummy_input["input_ids"], dummy_input["attention_mask"]),
            f              = onnx_output,
            opset_version  = 14,
            input_names    = ["input_ids", "attention_mask"],
            output_names   = ["logits"],
            dynamic_axes   = {
                "input_ids":      {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits":         {0: "batch_size"},
            },
            do_constant_folding = True,
        )

    onnx_size_mb = os.path.getsize(onnx_output) / (1024 * 1024)
    print(f"      ✅ ONNX diekspor: {onnx_output} ({onnx_size_mb:.1f} MB)")

    # ── Optional: simplify graph ──────────────────────────────────────────────
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim

            print("      Menyederhanakan ONNX graph dengan onnxsim...")
            model_onnx = onnx.load(onnx_output)
            model_simplified, check = onnxsim(model_onnx)
            if check:
                onnx.save(model_simplified, onnx_output)
                new_size = os.path.getsize(onnx_output) / (1024 * 1024)
                print(f"      ✅ ONNX disederhanakan: {onnx_size_mb:.1f} MB → {new_size:.1f} MB")
            else:
                print("      ⚠️  onnxsim: verifikasi gagal, menggunakan model asli.")
        except ImportError:
            print("      ⚠️  onnxsim tidak terinstall, lewati simplifikasi.")
            print("         Install dengan: pip install onnxsim")

    # ── Verifikasi ────────────────────────────────────────────────────────────
    if verify:
        print(f"\n[3/3] Memverifikasi ONNX model...")
        try:
            import onnxruntime as ort
        except ImportError:
            print("      ⚠️  onnxruntime tidak terinstall, lewati verifikasi.")
            print("         Install dengan: pip install onnxruntime>=1.16.0")
            return onnx_output

        session = ort.InferenceSession(
            onnx_output,
            providers=["CPUExecutionProvider"],
        )
        input_names  = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        print(f"      ONNX inputs : {input_names}")
        print(f"      ONNX outputs: {output_names}")

        # Bandingkan logits PyTorch vs ONNX
        feed = {
            "input_ids":      dummy_input["input_ids"].numpy().astype(np.int64),
            "attention_mask": dummy_input["attention_mask"].numpy().astype(np.int64),
        }
        with torch.no_grad():
            pt_logits = pt_model(**dummy_input).logits.numpy()
        onnx_logits = session.run(None, feed)[0]
        max_diff    = float(np.max(np.abs(pt_logits - onnx_logits)))

        print(f"      Max diff PyTorch vs ONNX: {max_diff:.8f}")
        if max_diff < 1e-3:
            print(f"      ✅ Output konsisten (diff < 1e-3)")
        else:
            print(f"      ⚠️  Diff tinggi ({max_diff:.6f}) — periksa opset/dynamic axes")

        # Ukur latensi CPU
        import time
        warmup_runs = 3
        bench_runs  = 10
        for _ in range(warmup_runs):
            session.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(bench_runs):
            session.run(None, feed)
        avg_ms = (time.perf_counter() - t0) / bench_runs * 1000
        print(f"      ✅ Rata-rata latency ONNX (CPU): {avg_ms:.1f}ms per sample")

    return onnx_output


def main():
    args   = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("CkCk — Export ke ONNX + Simpan Tokenizer Lokal")
    print("=" * 60)

    # Step 1: Simpan tokenizer
    tokenizer_path = save_tokenizer(config)

    # Step 2: Export ONNX
    onnx_path = export_onnx(config, verify=args.verify or True, simplify=args.simplify)

    print("\n" + "=" * 60)
    print("✅ Export selesai!")
    print(f"   Tokenizer : {tokenizer_path}")
    print(f"   ONNX model: {onnx_path}")
    print("\nSekarang jalankan inference.ipynb — USE_MOCK akan otomatis False.")
    print("=" * 60)


if __name__ == "__main__":
    main()
