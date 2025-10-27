# app/ocr_utils.py
import os, sys, traceback, shutil
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image
from typing import List

def _log_err(msg: str, exc: Exception | None = None):
    print(f"[OCR] {msg}", file=sys.stderr)
    if exc:
        traceback.print_exc()

def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None

def pdf_to_text_if_digital(pdf_path: str) -> str:
    """Use pypdf for selectable-text PDFs; return '' if not digital."""
    try:
        reader = PdfReader(pdf_path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n\n".join(pages).strip()
        return text if len(text) > 50 else ""
    except Exception as e:
        _log_err(f"pypdf failed for {pdf_path}", e)
        return ""

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)

def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    if not _tesseract_available():
        raise RuntimeError("Tesseract is not installed on PATH in this runtime.")
    return pytesseract.image_to_string(img, lang=lang)

def ocr_file(path: str, lang: str = "eng") -> str:
    """Robust OCR: .txt → return; .pdf → pypdf then OCR; images → OCR; else ''. """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        if ext == ".pdf":
            digital = pdf_to_text_if_digital(path)
            if digital:
                return digital
            # fallback to OCR only if available
            if not _tesseract_available():
                _log_err("Tesseract missing; cannot OCR scanned PDF. Install via apt.txt.")
                return ""
            pages = pdf_to_images(path)
            texts = [ocr_image(p, lang=lang) for p in pages]
            return "\n\n".join(texts)

        if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
            if not _tesseract_available():
                _log_err("Tesseract missing; cannot OCR image. Install via apt.txt.")
                return ""
            img = Image.open(path)
            return ocr_image(img, lang=lang)

        _log_err(f"Unsupported file type: {ext}")
        return ""
    except Exception as e:
        _log_err(f"OCR failed for {path}", e)
        return ""
