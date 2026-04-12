
import logging
import numpy as np
from typing import Union, Optional

logger = logging.getLogger(__name__)

# ── Keyword yang memicu prediksi HOAX pada mock ───────────────────────────────
# Gunakan ini dalam unit test untuk membuat assertion deterministik.

_MOCK_HOAX_KEYWORDS = {
    "hoax", "hoaks", "breaking", "viral", "sebarkan", "microchip", "5g",
    "vaksin berbahaya", "dihapus", "awas", "bahaya", "forward", "share",
}

_MOCK_HIGH_CONFIDENCE  = 0.9200
_MOCK_LOW_CONFIDENCE   = 0.8500

LABEL_MAP = {0: "VALID", 1: "HOAX"}

# Simulasikan "model ONNX" dan "tokenizer"
_MOCK_MODEL_OBJ = object()   # placeholder bukan None, agar isinstance check aman


# ═══════════════════════════════════════════════════════════════════════════════
# 1. load_models (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(config: dict) -> dict:
    logger.warning("[MOCK] load_models dipanggil — menggunakan mock model.")
    print("[MOCK] ⚠️  Menggunakan MOCK model — bukan model ONNX nyata.")

    max_length = config.get("model", {}).get("max_length", 256)

    return {
        "session":        _MOCK_MODEL_OBJ,
        "tokenizer":      _MOCK_MODEL_OBJ,
        "max_length":     max_length,
        "input_names":    ["input_ids", "attention_mask"],
        "output_names":   ["logits"],
        "model_path":     "[MOCK] tidak ada file",
        "tokenizer_path": "[MOCK] tidak ada file",
        "_is_mock":       True,           # penanda mode mock
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. prepare_input (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_input(
    input_type: str,
    source_data: Union[str, bytes],
    caption_fallback: Optional[str] = None,
    ocr_min_chars: int = 15,
    ocr_languages: Optional[list] = None,
) -> dict:
    logger.warning(f"[MOCK] prepare_input dipanggil — input_type='{input_type}'")
    input_type = input_type.strip().lower()

    # ── Caption ────────────────────────────────────────────────────────────────
    if input_type == "caption":
        if not isinstance(source_data, str):
            source_data = str(source_data)
        return {
            "text":          source_data,
            "source":        "caption",
            "ocr_result":    None,
            "fallback_used": False,
        }

    # ── Frame (simulasi OCR) ───────────────────────────────────────────────────
    elif input_type == "frame":
        ocr_text = _simulate_ocr(source_data, mode="frame")
        return _apply_fallback_logic(ocr_text, "ocr_frame", caption_fallback, ocr_min_chars)

    # ── Video (simulasi OCR multi-frame) ──────────────────────────────────────
    elif input_type == "video":
        ocr_text = _simulate_ocr(source_data, mode="video")
        return _apply_fallback_logic(ocr_text, "ocr_video", caption_fallback, ocr_min_chars)

    else:
        raise ValueError(
            f"[MOCK] input_type tidak dikenal: '{input_type}'. "
            "Gunakan: 'caption', 'frame', atau 'video'."
        )


def _simulate_ocr(source: Union[str, bytes], mode: str) -> str:
    if isinstance(source, bytes):
        if len(source) < 20:
            # Simulasi OCR gagal (gambar terlalu kecil / corrupt)
            logger.warning(f"[MOCK] Simulasi OCR gagal — bytes terlalu pendek ({len(source)} byte).")
            return ""
        # Simulasi OCR berhasil
        mock_text = f"[MOCK OCR {mode}] Teks hasil simulasi OCR dari input bytes."
    else:
        # Path file — anggap berhasil
        fname = str(source)
        mock_text = f"[MOCK OCR {mode}] Teks OCR simulasi dari file: {fname}"

    logger.info(f"[MOCK] OCR simulasi menghasilkan {len(mock_text)} karakter.")
    return mock_text


def _apply_fallback_logic(
    ocr_text: str,
    ocr_source: str,
    fallback: Optional[str],
    min_chars: int,
) -> dict:
    """Sama dengan logika di inferencer.py."""
    ocr_clean = ocr_text.strip() if ocr_text else ""

    if len(ocr_clean) >= min_chars:
        return {
            "text":          ocr_clean,
            "source":        ocr_source,
            "ocr_result":    ocr_clean,
            "fallback_used": False,
        }

    logger.warning(
        f"[MOCK] OCR {len(ocr_clean)} karakter < {min_chars}. Fallback ke caption."
    )

    if not fallback:
        raise ValueError(
            f"[MOCK] OCR tidak memadai dan caption_fallback tidak disediakan."
        )

    return {
        "text":          fallback.strip(),
        "source":        "fallback",
        "ocr_result":    ocr_clean,
        "fallback_used": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. run_classifier (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

def run_classifier(clean_text: str, models: dict) -> dict:
    logger.warning("[MOCK] run_classifier dipanggil.")

    if not clean_text or not clean_text.strip():
        raise ValueError("[MOCK] Input teks kosong.")

    text_lower = clean_text.lower()
    is_hoax    = any(kw in text_lower for kw in _MOCK_HOAX_KEYWORDS)

    if is_hoax:
        # Logits condong ke index 1 (HOAX)
        logits = np.array([[-2.1, 3.5]], dtype=np.float32)
    else:
        # Logits condong ke index 0 (VALID)
        logits = np.array([[3.2, -1.8]], dtype=np.float32)

    return {
        "logits":     logits,
        "input_text": clean_text,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. compute_confidence (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confidence(classifier_output: dict) -> dict:
    logits = classifier_output["logits"]   # shape: (1, num_labels)

    logits_flat = logits[0].astype(np.float64)
    shifted     = logits_flat - np.max(logits_flat)
    exp_vals    = np.exp(shifted)
    probs       = exp_vals / np.sum(exp_vals)

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label      = LABEL_MAP.get(pred_idx, f"LABEL_{pred_idx}")

    return {
        "label":      label,
        "label_id":   pred_idx,
        "confidence": round(confidence, 4),
        "probabilities": {
            LABEL_MAP.get(i, f"LABEL_{i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# run_ckck_inference (Mock convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ckck_inference(
    raw_input: Union[str, bytes],
    input_type: str,
    models: dict,
    pii_filter,
    preprocessor,
    support_scorer=None,
    caption_fallback: Optional[str] = None,
    ocr_min_chars: int = 15,
    verbose: bool = False,
) -> dict:

    import time
    start = time.time()

    prepared       = prepare_input(input_type, raw_input, caption_fallback, ocr_min_chars)
    raw_text       = prepared["text"]

    pii_result     = pii_filter.filter(raw_text)
    safe_text      = pii_result["filtered_text"]

    clean_text     = preprocessor.clean(safe_text, normalize_slang=True)

    classifier_out = run_classifier(clean_text, models)
    confidence_out = compute_confidence(classifier_out)

    support_data = {}
    if support_scorer is not None:
        support = support_scorer(clean_text)
        support_data = {
            "support_score":  support.support_score,
            "risk_level":     support.risk_level,
            "support_detail": support.category_hits,
            "explanation":    support.explanation,
        }

    elapsed_ms = round((time.time() - start) * 1000, 2)

    result = {
        "input_type":        input_type,
        "input_source":      prepared["source"],
        "ocr_result":        prepared["ocr_result"],
        "fallback_used":     prepared["fallback_used"],
        "raw_text":          raw_text,
        "pii_filtered_text": safe_text,
        "cleaned_text":      clean_text,
        "prediction":        confidence_out["label"],
        "confidence":        confidence_out["confidence"],
        "probabilities":     confidence_out["probabilities"],
        "pii_detected":      pii_result["pii_count"],
        "pii_details":       pii_result["details"],
        "inference_time_ms": elapsed_ms,
        "_mock":             True,
        **support_data,
    }

    if verbose:
        icon = "🔴" if confidence_out["label"] == "HOAX" else "🟢"
        print("━" * 55)
        print(f"[MOCK] 📥 Input ({input_type}): {str(raw_input)[:60]}")
        if prepared["fallback_used"]:
            print("[MOCK] ⚠️  Fallback digunakan")
        if pii_result["pii_count"] > 0:
            print(f"[MOCK] 🔒 PII: {pii_result['pii_count']} item disensor")
        print(f"[MOCK] {icon} Prediksi: {confidence_out['label']} ({confidence_out['confidence']*100:.1f}%)")
        print(f"[MOCK] ⏱️  Waktu   : {elapsed_ms}ms")
        print("━" * 55)

    return result


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.pii_filter import PIIFilter
    from src.preprocessing import TextPreprocessor
    from src.manipulative_detector import compute_support_score

    print("=" * 55)
    print("MOCK INFERENCER — Self-Test")
    print("=" * 55)

    pii   = PIIFilter()
    prep  = TextPreprocessor()
    config = {"model": {"max_length": 256}, "inference": {}}
    mods  = load_models(config)

    tests = [
        ("caption", "Pemerintah mengumumkan kebijakan ekonomi baru.", "VALID"),
        ("caption", "BREAKING!! Vaksin mengandung microchip 5G!! SEBARKAN!!", "HOAX"),
        ("caption", "NIK saya 3201234506780001, segera forward ke semua!", "HOAX"),
    ]

    all_pass = True
    for input_type, text, expected in tests:
        result = run_ckck_inference(
            raw_input     = text,
            input_type    = input_type,
            models        = mods,
            pii_filter    = pii,
            preprocessor  = prep,
            support_scorer = compute_support_score,
            verbose       = False,
        )
        ok = result["prediction"] == expected
        if not ok:
            all_pass = False
        status = "✅" if ok else f"❌ (expected {expected})"
        print(f"\n[{status}] '{text[:60]}'")
        print(f"  Prediksi : {result['prediction']} ({result['confidence']:.0%})")
        print(f"  PII      : {result['pii_detected']} item")

    print("\n" + "=" * 55)
    print(f"{'Semua test lulus ✅' if all_pass else 'Ada test gagal ❌'}")
    print("=" * 55)
