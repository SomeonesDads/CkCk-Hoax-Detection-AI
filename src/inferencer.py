import os
import io
import logging
import numpy as np
from typing import Union, Optional

logger = logging.getLogger(__name__)

INPUT_TYPE_CAPTION = "caption"   # teks langsung dari caption / judul postingan
INPUT_TYPE_FRAME   = "frame"     # gambar statis (JPG, PNG, WEBP, dll)
INPUT_TYPE_VIDEO   = "video"     # file video (MP4, AVI, MKV, dll)

LABEL_MAP    = {0: "VALID", 1: "HOAX"}
OCR_MIN_CHARS = 15               # minimal karakter OCR agar tidak fallback

_ocr_reader_cache: dict = {}

def load_models(config: dict) -> dict:

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "[load_models] onnxruntime tidak terinstall.\n"
            "Install dengan: pip install onnxruntime>=1.16.0"
        )

    inference_cfg  = config["inference"]
    model_cfg      = config["model"]

    model_path     = inference_cfg.get("model_path", "models/indobert_classifier.onnx")
    tokenizer_path = inference_cfg.get("tokenizer_path", "models/tokenizer")
    max_length     = model_cfg.get("max_length", 256)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[load_models] File ONNX tidak ditemukan: '{model_path}'\n"
            "Pastikan sudah:\n"
            "  1. Menjalankan training.ipynb sampai selesai\n"
            "  2. Menjalankan cell 'Export ke ONNX' di training.ipynb"
        )

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"[load_models] Direktori tokenizer tidak ditemukan: '{tokenizer_path}'\n"
            "Pastikan setelah fine-tuning, tokenizer sudah disimpan dengan:\n"
            "  tokenizer.save_pretrained('models/tokenizer')"
        )

    # ── Load ONNX Session ─────────────────────────────────────────────────────
    logger.info(f"[load_models] Loading ONNX session dari: {model_path}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2   # hemat resource, deterministik
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],     # ← Constraint B-2: CPU only
    )

    # Nama input ONNX (biasanya: input_ids, attention_mask, token_type_ids)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info(f"[load_models] ONNX inputs : {input_names}")
    logger.info(f"[load_models] ONNX outputs: {output_names}")

    # ── Load Tokenizer (lokal, tanpa download) ────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "[load_models] transformers tidak terinstall.\n"
            "Install dengan: pip install transformers>=4.36.0"
        )

    logger.info(f"[load_models] Loading tokenizer dari: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,          # ← Constraint B-2: offline
    )

    models = {
        "session":       session,
        "tokenizer":     tokenizer,
        "max_length":    max_length,
        "input_names":   input_names,
        "output_names":  output_names,
        "model_path":    model_path,
        "tokenizer_path": tokenizer_path,
    }

    logger.info("[load_models] ✅ Model dan tokenizer berhasil dimuat (offline, CPU-only)")
    return models

def prepare_input(
    input_type: str,
    source_data: Union[str, bytes],
    caption_fallback: Optional[str] = None,
    ocr_min_chars: int = OCR_MIN_CHARS,
    ocr_languages: Optional[list] = None,
) -> dict:
   
    if ocr_languages is None:
        ocr_languages = ["id", "en"]

    input_type = input_type.strip().lower()

    if input_type == INPUT_TYPE_CAPTION:
        if not isinstance(source_data, str):
            source_data = str(source_data)
        if not source_data.strip():
            logger.warning("[prepare_input] Caption kosong diterima.")
        return {
            "text":          source_data,
            "source":        "caption",
            "ocr_result":    None,
            "fallback_used": False,
        }

    elif input_type == INPUT_TYPE_FRAME:
        ocr_text = _run_ocr_on_image(source_data, languages=ocr_languages)
        return _apply_fallback_logic(
            ocr_text   = ocr_text,
            ocr_source = "ocr_frame",
            fallback   = caption_fallback,
            min_chars  = ocr_min_chars,
        )

    elif input_type == INPUT_TYPE_VIDEO:
        ocr_text = _run_ocr_on_video(source_data, languages=ocr_languages)
        return _apply_fallback_logic(
            ocr_text   = ocr_text,
            ocr_source = "ocr_video",
            fallback   = caption_fallback,
            min_chars  = ocr_min_chars,
        )

    else:
        raise ValueError(
            f"[prepare_input] input_type tidak dikenal: '{input_type}'.\n"
            f"Gunakan salah satu dari: 'caption', 'frame', 'video'."
        )



def _run_ocr_on_image(source: Union[str, bytes], languages: list) -> str:

    reader = _get_ocr_reader(languages)

    if isinstance(source, bytes):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "[inferencer] Pillow tidak terinstall.\n"
                "Install dengan: pip install Pillow>=10.0.0"
            )
        image   = Image.open(io.BytesIO(source)).convert("RGB")
        img_arr = np.array(image)
        results = reader.readtext(img_arr, detail=0, paragraph=True)

    else:
        if not os.path.exists(str(source)):
            raise FileNotFoundError(
                f"[prepare_input] File gambar tidak ditemukan: '{source}'"
            )
        results = reader.readtext(str(source), detail=0, paragraph=True)

    text = " ".join(results).strip()
    logger.info(
        f"[prepare_input] OCR gambar selesai — "
        f"{len(text)} karakter: '{text[:80]}{'...' if len(text) > 80 else ''}'"
    )
    return text


def _run_ocr_on_video(video_path: str, languages: list) -> str:
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "[inferencer] OpenCV tidak terinstall.\n"
            "Install dengan: pip install opencv-python-headless>=4.8.0"
        )

    if not os.path.exists(str(video_path)):
        raise FileNotFoundError(
            f"[prepare_input] File video tidak ditemukan: '{video_path}'"
        )

    logger.info(f"[prepare_input] Mengekstrak frame dari video: {video_path}")

    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        logger.warning("[prepare_input] Video kosong atau tidak bisa dibaca.")
        return ""

    sample_ratios   = [0.10, 0.30, 0.50, 0.70, 0.90]
    sample_positions = sorted(set(
        max(0, min(int(total_frames * r), total_frames - 1))
        for r in sample_ratios
    ))

    reader     = _get_ocr_reader(languages)
    all_texts  = []

    for frame_idx in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = reader.readtext(frame_rgb, detail=0, paragraph=True)
        frame_text = " ".join(results).strip()

        if frame_text:
            all_texts.append(frame_text)
            logger.debug(
                f"[prepare_input] Frame {frame_idx}: '{frame_text[:60]}'"
            )

    cap.release()

    seen, unique = set(), []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    combined = " ".join(unique).strip()
    logger.info(
        f"[prepare_input] OCR video selesai — "
        f"{len(combined)} karakter dari {len(unique)} frame unik."
    )
    return combined


def _apply_fallback_logic(
    ocr_text: str,
    ocr_source: str,
    fallback: Optional[str],
    min_chars: int,
) -> dict:
    """
    Terapkan logika fallback: jika OCR < min_chars karakter, gunakan caption.
    """
    ocr_clean = ocr_text.strip() if ocr_text else ""

    if len(ocr_clean) >= min_chars:
        # OCR memadai — lanjutkan
        return {
            "text":          ocr_clean,
            "source":        ocr_source,
            "ocr_result":    ocr_clean,
            "fallback_used": False,
        }

    # OCR tidak memadai
    logger.warning(
        f"[prepare_input] OCR hanya menghasilkan {len(ocr_clean)} karakter "
        f"(threshold: {min_chars}). Menggunakan caption fallback."
    )

    if not fallback:
        raise ValueError(
            f"[prepare_input] OCR tidak memadai ({len(ocr_clean)} karakter dari "
            f"threshold {min_chars}) dan caption_fallback tidak disediakan.\n"
            "Sertakan caption_fallback= saat memanggil prepare_input()."
        )

    return {
        "text":          fallback.strip(),
        "source":        "fallback",
        "ocr_result":    ocr_clean,
        "fallback_used": True,
    }


# ── Singleton EasyOCR Reader ──────────────────────────────────────────────────

def _get_ocr_reader(languages: list):
    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "[inferencer] EasyOCR tidak terinstall.\n"
            "Install dengan: pip install easyocr>=1.7.1"
        )

    key = tuple(sorted(languages))
    if key not in _ocr_reader_cache:
        logger.info(f"[OCR] Inisialisasi EasyOCR untuk bahasa: {languages} (pertama kali)")
        _ocr_reader_cache[key] = easyocr.Reader(
            languages,
            gpu=False,       # ← constraint: CPU only
            verbose=False,
        )
        logger.info("[OCR] EasyOCR siap digunakan.")

    return _ocr_reader_cache[key]


def run_classifier(clean_text: str, models: dict) -> dict:
    if not clean_text or not clean_text.strip():
        raise ValueError(
            "[run_classifier] Input teks kosong setelah preprocessing. "
            "Pastikan prepare_input() dan PII filter menghasilkan teks yang valid."
        )

    tokenizer   = models["tokenizer"]
    session     = models["session"]
    max_length  = models["max_length"]
    input_names = models["input_names"]

    # ── Tokenisasi → numpy (BUKAN PyTorch tensor "pt") ────────────────────────
    encoding = tokenizer(
        clean_text,
        max_length      = max_length,
        padding         = "max_length",
        truncation      = True,
        return_tensors  = "np",        # ← numpy, constraint ONNX
    )

    # Buat feed dict — hanya sertakan input yang dikenal model ONNX
    feed_dict: dict[str, np.ndarray] = {}
    for name in input_names:
        if name in encoding:
            feed_dict[name] = encoding[name].astype(np.int64)

    if "input_ids" not in feed_dict:
        raise RuntimeError(
            "[run_classifier] 'input_ids' tidak ada dalam output tokenizer. "
            "Periksa kompatibilitas tokenizer dengan model ONNX."
        )

    logger.debug(f"[run_classifier] Feed dict keys: {list(feed_dict.keys())}")
    logger.debug(f"[run_classifier] input_ids shape: {feed_dict['input_ids'].shape}")

    # ── ONNX Inference ────────────────────────────────────────────────────────
    ort_outputs = session.run(None, feed_dict)

    # Output pertama ONNX biasanya logits, shape: (batch_size, num_labels)
    logits = ort_outputs[0]    # np.ndarray  shape: (1, 2)

    return {
        "logits":     logits,
        "input_text": clean_text,
    }


def compute_confidence(classifier_output: dict) -> dict:
    logits = classifier_output["logits"]   # shape: (1, num_labels)


    logits_flat = logits[0].astype(np.float64)   # shape: (num_labels,)
    shifted     = logits_flat - np.max(logits_flat)
    exp_vals    = np.exp(shifted)
    probs       = exp_vals / np.sum(exp_vals)    # shape: (num_labels,)

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


def run_ckck_inference(
    raw_input: Union[str, bytes],
    input_type: str,
    models: dict,
    pii_filter,
    preprocessor,
    support_scorer=None,
    caption_fallback: Optional[str] = None,
    ocr_min_chars: int = OCR_MIN_CHARS,
    verbose: bool = False,
) -> dict:
    import time
    start = time.time()

    # ── Step 1: Prepare Input ──────────────────────────────────────────────────
    prepared = prepare_input(
        input_type       = input_type,
        source_data      = raw_input,
        caption_fallback = caption_fallback,
        ocr_min_chars    = ocr_min_chars,
    )
    raw_text = prepared["text"]

    # ── Step 2: PII Filter ────────────────────────────────────────────────────
    pii_result = pii_filter.filter(raw_text)
    safe_text  = pii_result["filtered_text"]

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    clean_text = preprocessor.clean(safe_text, normalize_slang=True)

    # ── Step 4: Classify ──────────────────────────────────────────────────────
    classifier_out = run_classifier(clean_text, models)
    confidence_out = compute_confidence(classifier_out)

    # ── Step 5: Support Score (opsional) ──────────────────────────────────────
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
        **support_data,
    }

    if verbose:
        icon = "🔴" if confidence_out["label"] == "HOAX" else "🟢"
        print("━" * 55)
        label_disp = str(raw_input)[:70]
        print(f"📥 Input ({input_type}): {label_disp}")
        if prepared["fallback_used"]:
            print(f"⚠️  OCR tidak memadai ({len(prepared['ocr_result'])} char) → fallback ke caption")
        if pii_result["pii_count"] > 0:
            print(f"🔒 PII: {pii_result['pii_count']} item disensor")
        print(
            f"{icon} Prediksi   : {confidence_out['label']} "
            f"({confidence_out['confidence']*100:.1f}%)"
        )
        if support_data:
            print(f"📊 Risiko    : {support_data.get('risk_level', '-')}")
        print(f"⏱️  Waktu     : {elapsed_ms}ms")
        print("━" * 55)

    return result


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick self-test menggunakan mock inferencer agar bisa berjalan
    tanpa model ONNX yang sudah jadi.
    """
    from src.mock_inferencer import (
        load_models as mock_load,
        prepare_input as mock_prepare,
        run_classifier as mock_run,
        compute_confidence as mock_conf,
    )

    print("=" * 55)
    print("INFERENCER — Self-Test (Mock Mode)")
    print("=" * 55)

    result = mock_prepare("caption", "BREAKING!! Vaksin mengandung microchip 5G!!")
    assert result["source"] == "caption"
    assert result["fallback_used"] == False
    print(f"✅ prepare_input(caption): '{result['text'][:50]}'")

    result = mock_prepare("frame", b"tiny", caption_fallback="Caption darurat")
    assert result["source"] == "fallback"
    assert result["fallback_used"] == True
    print(f"✅ prepare_input(frame, fallback): source='{result['source']}'")

    dummy_logits = np.array([[0.3, 2.1]])
    conf = mock_conf({"logits": dummy_logits})
    assert conf["label"] in ("HOAX", "VALID")
    assert 0.0 <= conf["confidence"] <= 1.0
    print(f"✅ compute_confidence: {conf['label']} ({conf['confidence']:.2%})")

    print("\n✅ Semua self-test lulus.\n")
