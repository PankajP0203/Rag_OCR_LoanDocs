import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import List

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)

def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(img, lang=lang)

def ocr_file(path: str, lang: str = "eng") -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in [".pdf"]:
        pages = pdf_to_images(path)
        texts = [ocr_image(p, lang=lang) for p in pages]
        return "\n\n".join(texts)
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        img = Image.open(path)
        return ocr_image(img, lang=lang)
    raise ValueError(f"Unsupported file type: {ext}")
