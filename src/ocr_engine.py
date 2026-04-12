"""
OCR Engine
=============
Extracts text from images for hoax detection pipeline.
Uses pytesseract (primary) with EasyOCR fallback.
Falls back to image caption/filename if OCR result < 15 characters.

Constraint: Runs 100% offline — no API calls.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR text extraction."""
    text: str                   # Extracted text
    source: str                 # 'pytesseract', 'easyocr', or 'fallback'
    confidence: float           # Estimated confidence (0-1)
    char_count: int             # Number of characters extracted
    used_fallback: bool         # Whether fallback was used


# Minimum character threshold — below this, OCR result is unreliable
MIN_OCR_CHARS = 15


class OCREngine:
    """
    OCR text extraction for image-based hoax detection.
    
    Supports pytesseract (Tesseract OCR) as primary engine with
    EasyOCR as fallback. If extracted text is too short (< 15 chars),
    falls back to any provided caption text.
    
    Usage:
        ocr = OCREngine()
        result = ocr.extract("path/to/screenshot.jpg")
        print(result.text)
    
    Installation:
        - pytesseract: pip install pytesseract + install Tesseract binary
        - easyocr: pip install easyocr (optional fallback)
    """

    def __init__(self, lang: str = "ind", use_easyocr_fallback: bool = True):
        """
        Initialize OCR Engine.
        
        Args:
            lang: OCR language. 'ind' for Indonesian, 'eng' for English.
            use_easyocr_fallback: Whether to try EasyOCR if pytesseract fails.
        """
        self.lang = lang
        self.use_easyocr_fallback = use_easyocr_fallback
        self._pytesseract = None
        self._easyocr_reader = None
        self._pil_image = None

        # Try to import PIL
        try:
            from PIL import Image
            self._pil_image = Image
        except ImportError:
            print("[WARN] Pillow not installed. Install with: pip install Pillow")

        # Try to import pytesseract
        try:
            import pytesseract
            self._pytesseract = pytesseract
            print("[INFO] OCR Engine: pytesseract available ✅")
        except ImportError:
            print("[WARN] pytesseract not installed. Install with: pip install pytesseract")
            print("       Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract")

        # Try to import EasyOCR (lazy — only when needed)
        if use_easyocr_fallback:
            try:
                import easyocr
                print("[INFO] OCR Engine: EasyOCR available as fallback ✅")
            except ImportError:
                print("[INFO] OCR Engine: EasyOCR not available (optional fallback)")
                self.use_easyocr_fallback = False

    def _extract_pytesseract(self, image_path: str) -> Optional[str]:
        """Extract text using pytesseract."""
        if self._pytesseract is None or self._pil_image is None:
            return None
        try:
            image = self._pil_image.open(image_path)
            text = self._pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            print(f"[WARN] pytesseract failed: {e}")
            return None

    def _extract_easyocr(self, image_path: str) -> Optional[str]:
        """Extract text using EasyOCR as fallback."""
        if not self.use_easyocr_fallback:
            return None
        try:
            import easyocr
            if self._easyocr_reader is None:
                # Initialize reader (lazy loading, first call may take a few seconds)
                lang_map = {"ind": "id", "eng": "en"}
                reader_lang = lang_map.get(self.lang, "id")
                self._easyocr_reader = easyocr.Reader([reader_lang], gpu=False)
            
            results = self._easyocr_reader.readtext(image_path)
            text = " ".join([result[1] for result in results])
            return text.strip()
        except Exception as e:
            print(f"[WARN] EasyOCR failed: {e}")
            return None

    def extract(self, image_path: str, caption: str = "") -> OCRResult:
        """
        Extract text from an image file.
        
        Tries pytesseract first, then EasyOCR, then falls back to caption.
        
        Args:
            image_path: Path to the image file.
            caption: Optional caption/alt-text to use as fallback.
            
        Returns:
            OCRResult with extracted text and metadata.
        """
        if not os.path.exists(image_path):
            return OCRResult(
                text=caption or "",
                source="fallback",
                confidence=0.0,
                char_count=len(caption),
                used_fallback=True,
            )

        # Attempt 1: pytesseract
        text = self._extract_pytesseract(image_path)
        if text and len(text) >= MIN_OCR_CHARS:
            return OCRResult(
                text=text,
                source="pytesseract",
                confidence=0.8,
                char_count=len(text),
                used_fallback=False,
            )

        # Attempt 2: EasyOCR fallback
        text_easyocr = self._extract_easyocr(image_path)
        if text_easyocr and len(text_easyocr) >= MIN_OCR_CHARS:
            return OCRResult(
                text=text_easyocr,
                source="easyocr",
                confidence=0.7,
                char_count=len(text_easyocr),
                used_fallback=False,
            )

        # Attempt 3: Fallback to caption
        # Use whatever OCR text we got (even if short), or caption
        best_text = text or text_easyocr or caption or ""
        return OCRResult(
            text=best_text if len(best_text) >= MIN_OCR_CHARS else (caption or best_text),
            source="fallback",
            confidence=0.3,
            char_count=len(best_text),
            used_fallback=True,
        )

    def is_available(self) -> bool:
        """Check if at least one OCR backend is available."""
        return self._pytesseract is not None or self.use_easyocr_fallback


if __name__ == "__main__":
    ocr = OCREngine()
    print(f"\nOCR available: {ocr.is_available()}")
    print("\nTo test with an image:")
    print("  result = ocr.extract('path/to/image.jpg')")
    print("  print(result.text)")
