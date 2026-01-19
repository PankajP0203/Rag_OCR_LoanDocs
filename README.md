# RAG-based OCR System for Loan Documents


## Overview
This project demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** system
for understanding loan documents (sanction letters, statements, scanned PDFs).

It combines **OCR**, **vector search**, and **LLM-based answer synthesis** to answer
natural language questions such as:
- “What is the sanctioned loan amount?”
- “What is the EMI and tenure?”
- “Summarize key sanction terms.”

## Architecture

PDF / Image / TXT
↓
OCR (Tesseract / pypdf)
↓
Text Cleaning & Chunking
↓
Sentence Embeddings (MiniLM)
↓
FAISS Vector Store
↓
Top-K Retrieval
↓
LLM Synthesis (GPT-4o-mini)
↓
Natural Language Answer


## Tech Stack

- **OCR**: Tesseract, pdf2image, pypdf
- **Embeddings**: sentence-transformers (MiniLM)
- **Vector DB**: FAISS
- **LLM**: OpenAI GPT-4o-mini (via API)
- **Backend**: Python
- **UI**: Gradio
- **Deployment**: Hugging Face Spaces

## Features

- Supports scanned PDFs, images, and text files
- PII redaction before indexing
- Semantic search over document chunks
- Optional LLM-based answer synthesis
- Model selection via UI
- CPU-only, privacy-friendly deployment

## Running Locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxxx
python app_ui.py

Deployment
The app is deployed on Hugging Face Spaces using Gradio.
System dependencies are installed via apt.txt.

Use Case
Designed as a demo for FinTech / Lending / AI Product interviews,
showing how OCR + RAG + LLMs can automate document understanding workflows.

Author
Pankaj Panwar