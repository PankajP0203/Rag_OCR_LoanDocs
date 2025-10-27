# app/ocr_utils.py
import os
import sys
import traceback
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image
from typing import List

def log_err(msg: str, exc: Exception = None):
    print(f"[OCR] {msg}", file=sys.stderr)
    if exc:
        traceback.print_exc()

def pdf_to_text_if_digital(pdf_path: str) -> str:
    """Extract selectable text from PDFs (no OCR). Returns '' if not digital."""
    try:
        reader = PdfReader(pdf_path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n\n".join(pages).strip()
        # treat as digital only if there is meaningful length
        return text if len(text) > 50 else ""
    except Exception as e:
        log_err(f"pypdf failed for {pdf_path}", e)
        return ""

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)

def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(img, lang=lang)

def ocr_file(path: str, lang: str = "eng") -> str:
    """Robust OCR: try digital PDF text; else OCR pages; supports images & .txt."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        if ext == ".pdf":
            # 1) Digital text first (fast, avoids Poppler/Tesseract for native PDFs)
            digital = pdf_to_text_if_digital(path)
            if digital:
                return digital

            # 2) Fall back to OCR via pdf2image + tesseract
            pages = pdf_to_images(path)
            texts = [ocr_image(p, lang=lang) for p in pages]
            return "\n\n".join(texts)

        if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
            img = Image.open(path)
            return ocr_image(img, lang=lang)

        raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        log_err(f"OCR failed for {path}", e)
        # Best effort: return empty string so UI can surface a nice message
        return ""
