"""
Script pengujian lengkap CkCk Hoax Detection AI
================================================
Menguji 4 level pengujian dari ringan ke penuh:
  Level 1: Komponen individual (PII, Preprocessor, ManipulativeDetector)
  Level 2: Mock Inferencer pipeline penuh
  Level 3: compute_confidence dengan logits dummy
  Level 4: prepare_input (caption + fallback — tanpa OCR)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

PASS = 0
FAIL = 0

def ok(label):
    global PASS
    PASS += 1
    print(f"  ✅ {label}")

def fail(label, err):
    global FAIL
    FAIL += 1
    print(f"  ❌ {label}: {err}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# LEVEL 1 : Komponen individual
# ─────────────────────────────────────────────────────────────
section("LEVEL 1 — Komponen Individual")

# PII Filter
try:
    from src.pii_filter import PIIFilter
    pii = PIIFilter()

    r = pii.filter("NIK saya 3201234506780001 dan email budi@gmail.com")
    assert r["pii_count"] == 2
    ok(f"PIIFilter (2 PII): '{r['filtered_text'][:55]}'")

    r2 = pii.filter("Ini teks bersih tanpa PII apapun.")
    assert r2["pii_count"] == 0
    ok("PIIFilter (0 PII) tidak false-positive")

    r3 = pii.filter("Hubungi +6281234567890 untuk info transfer ke rekening BCA 1234567890123456")
    assert r3["pii_count"] == 2
    ok(f"PIIFilter (telepon + rekening): {r3['pii_count']} item")
except Exception as e:
    fail("PIIFilter", e)

# Preprocessor
try:
    from src.preprocessing import TextPreprocessor
    prep = TextPreprocessor()
    out = prep.clean("BREAKING!!! ini gak benar!! https://hoax.com #viral @beritahoax")
    assert "https" not in out
    assert "@beritahoax" not in out
    assert "#viral" not in out
    ok(f"TextPreprocessor: '{out}'")
except Exception as e:
    fail("TextPreprocessor", e)

# Manipulative Detector
try:
    from src.manipulative_detector import compute_support_score
    r_hoax = compute_support_score("BREAKING!! Vaksin mengandung microchip 5G!! SEBARKAN sebelum dihapus!!")
    assert r_hoax.risk_level == "TINGGI", f"Got {r_hoax.risk_level}"
    ok(f"ManipulativeDetector TINGGI: skor={r_hoax.support_score:.2f}")

    r_clean = compute_support_score("Pemerintah mengumumkan kebijakan ekonomi baru di sektor teknologi.")
    assert r_clean.risk_level == "RENDAH", f"Got {r_clean.risk_level}"
    ok(f"ManipulativeDetector RENDAH: skor={r_clean.support_score:.2f}")
except Exception as e:
    fail("ManipulativeDetector", e)


# ─────────────────────────────────────────────────────────────
# LEVEL 2 : compute_confidence (numpy softmax)
# ─────────────────────────────────────────────────────────────
section("LEVEL 2 — compute_confidence (Numpy Softmax)")

try:
    from src.mock_inferencer import compute_confidence

    # Logits condong ke HOAX (index 1)
    c = compute_confidence({"logits": np.array([[-2.1, 3.5]])})
    assert c["label"] == "HOAX", f"Expected HOAX got {c['label']}"
    assert 0.95 < c["confidence"] <= 1.0, f"Confidence too low: {c['confidence']}"
    ok(f"compute_confidence HOAX: confidence={c['confidence']:.4f}")

    # Logits condong ke VALID (index 0)
    c2 = compute_confidence({"logits": np.array([[3.2, -1.8]])})
    assert c2["label"] == "VALID", f"Expected VALID got {c2['label']}"
    assert c2["confidence"] > 0.9
    ok(f"compute_confidence VALID: confidence={c2['confidence']:.4f}")

    # Probabilitas harus sum ke 1.0
    total = sum(c["probabilities"].values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum = {total}"
    ok(f"Probabilities sum = {total:.8f} ≈ 1.0")
except Exception as e:
    fail("compute_confidence", e)


# ─────────────────────────────────────────────────────────────
# LEVEL 3 : prepare_input (caption & fallback)
# ─────────────────────────────────────────────────────────────
section("LEVEL 3 — prepare_input (Caption + Fallback)")

try:
    from src.mock_inferencer import prepare_input

    # Caption langsung
    r = prepare_input("caption", "Sri Mulyani menyebut guru itu beban negara, SEBARKAN!")
    assert r["source"] == "caption"
    assert r["fallback_used"] == False
    assert r["ocr_result"] is None
    ok(f"prepare_input(caption): source='{r['source']}'")

    # Frame bytes pendek → OCR gagal → fallback
    r2 = prepare_input("frame", b"tiny", caption_fallback="Caption postingan fallback")
    assert r2["source"] == "fallback"
    assert r2["fallback_used"] == True
    assert r2["text"] == "Caption postingan fallback"
    ok(f"prepare_input(frame, fallback): source='{r2['source']}'")

    # Frame bytes cukup panjang → OCR mock berhasil
    r3 = prepare_input("frame", b"x" * 30, caption_fallback="Fallback")
    assert r3["source"] == "ocr_frame"
    assert r3["fallback_used"] == False
    ok(f"prepare_input(frame, mock OCR): source='{r3['source']}'")

    # Video dengan fallback
    r4 = prepare_input("video", b"short", caption_fallback="Caption video TikTok")
    assert r4["source"] == "fallback"
    ok(f"prepare_input(video, fallback): source='{r4['source']}'")

    # Error jika fallback tidak disediakan tapi OCR gagal
    try:
        prepare_input("frame", b"x")   # bytes pendek, tanpa fallback
        fail("prepare_input (no fallback)", "Seharusnya raise ValueError")
    except ValueError:
        ok("prepare_input gagal dengan jelas jika OCR tidak memadai & no fallback")
except Exception as e:
    fail("prepare_input", e)


# ─────────────────────────────────────────────────────────────
# LEVEL 4 : Pipeline penuh via Mock
# ─────────────────────────────────────────────────────────────
section("LEVEL 4 — Pipeline Penuh (Mock Inferencer)")

try:
    from src.mock_inferencer import load_models, run_ckck_inference
    from src.pii_filter import PIIFilter
    from src.preprocessing import TextPreprocessor
    from src.manipulative_detector import compute_support_score

    pii_f  = PIIFilter()
    prep_f = TextPreprocessor()
    config = {"model": {"max_length": 256}, "inference": {}}
    mods   = load_models(config)

    test_cases = [
        # (input_text, expected_prediction)
        ("Pemerintah mengumumkan kebijakan ekonomi baru di sektor teknologi.", "VALID"),
        ("BREAKING!! Vaksin mengandung microchip 5G!! Bagikan sebelum dihapus!!", "HOAX"),
        ("AWAS bahaya!! SEBARKAN ke semua teman sekarang juga!!", "HOAX"),
        # Teks dengan PII — harus disensor lalu prediksi
        ("NIK 3201234506780001 hubungi +6281234567890 segera forward ke semua!", "HOAX"),
    ]

    for text, expected in test_cases:
        r = run_ckck_inference(
            raw_input      = text,
            input_type     = "caption",
            models         = mods,
            pii_filter     = pii_f,
            preprocessor   = prep_f,
            support_scorer = compute_support_score,
            verbose        = False,
        )
        passed = r["prediction"] == expected
        status = "✅" if passed else "❌"
        label = f"'{text[:55]}...'" if len(text) > 55 else f"'{text}'"
        if passed:
            ok(f"{r['prediction']:5s} ({r['confidence']:.0%}) | PII={r['pii_detected']} | {label}")
        else:
            fail(f"Pipeline {label}", f"Expected {expected}, got {r['prediction']}")

except Exception as e:
    fail("Pipeline Mock", e)


# ─────────────────────────────────────────────────────────────
# LEVEL 5 : Cek onnxruntime import
# ─────────────────────────────────────────────────────────────
section("LEVEL 5 — Verifikasi Library ONNX Runtime")

try:
    import onnxruntime as ort
    ok(f"onnxruntime {ort.__version__} terinstall")
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers
    ok(f"CPUExecutionProvider tersedia: {providers}")
except ImportError:
    fail("onnxruntime", "Belum terinstall — jalankan: pip install onnxruntime")
except Exception as e:
    fail("onnxruntime check", e)


# ─────────────────────────────────────────────────────────────
# Hasil Akhir
# ─────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*55}")
print(f"  HASIL: {PASS}/{total} test lulus")
if FAIL == 0:
    print("  🎉 Semua test LULUS! Kode siap diintegrasikan.")
else:
    print(f"  ⚠️  {FAIL} test GAGAL — periksa output di atas.")
print(f"{'='*55}")
