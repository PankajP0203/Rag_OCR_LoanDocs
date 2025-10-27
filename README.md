---
title: RAG OCR for Loan Sanction Letters
emoji: 🧾
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app_ui.py
pinned: false
license: mit
---

# 🧾 RAG-based OCR Demo (IIFL)

This Space demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for OCR-based document intelligence.

### 🔹 Features
- **OCR ingestion** for PDFs / images using Tesseract + Poppler  
- **Chunking & embedding** via `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector search** using FAISS  
- **Question answering** via retrieved context  
- **Optional LLM synthesis** if `OPENAI_API_KEY` is set in Space secrets  
- **Field extraction** (Sanctioned Amount, EMI, ROI, Tenure, etc.)

### 🚀 Run locally
```bash
git clone https://huggingface.co/spaces/PankajPanwar1101/rag_ocr_iifl
cd rag_ocr_iifl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app_ui.py
