# app_ui.py
import os, uuid
import gradio as gr

# Reuse your pipeline
from app.ocr_utils import ocr_file
from app.rag import RAGPipeline
try:
    from app.extractors import extract_fields
except Exception:
    extract_fields = None

STORE_DIR = os.path.abspath(os.getenv("STORE_DIR", "store"))
os.makedirs(STORE_DIR, exist_ok=True)
pipeline = RAGPipeline(STORE_DIR)

def ingest_file(file, lang="eng", redact=True):
    if file is None:
        return "Please upload a PDF/image/TXT.", None
    text = ocr_file(file.name, lang=lang)
    doc_id = f"{os.path.basename(file.name)}-{uuid.uuid4().hex[:8]}"
    chunks, tokens = pipeline.ingest_text(text, doc_id=doc_id, redact=redact)
    return f"Ingested: {doc_id} | chunks={chunks} | tokens~{tokens}", text

def ask_query(query, k=5, use_llm=False):
    if not query.strip():
        return "Type a question.", "", "", ""
    answer, contexts, scores = pipeline.answer(query, k=int(k), use_llm=use_llm)
    joined = "\n\n---\n\n".join([c for c in contexts if c])
    fields = None
    if extract_fields and joined:
        fields = extract_fields(joined)
    return answer, joined, str([round(float(s), 4) for s in scores]), fields or {}

with gr.Blocks(title="RAG OCR Demo (IIFL)") as demo:
    gr.Markdown("### RAG-based OCR (Demo)\nUpload a **PDF/Image/TXT**, then ask questions. *(All on CPU, privacy-friendly)*")

    with gr.Tab("1) Ingest"):
        with gr.Row():
            file = gr.File(label="Upload PDF / Image / TXT", file_count="single", type="filepath")
            lang = gr.Dropdown(["eng","eng+hin"], value="eng", label="OCR language")
            redact = gr.Checkbox(value=True, label="Redact common PII before indexing")
        ingest_btn = gr.Button("Ingest")
        ingest_status = gr.Textbox(label="Status", lines=2)
        preview_text = gr.Textbox(label="Extracted text (first 1000 chars)", lines=10)
        def _preview_wrap(file, lang, redact):
            msg, text = ingest_file(file, lang, redact)
            short = (text or "")[:1000]
            return msg, short
        ingest_btn.click(_preview_wrap, inputs=[file, lang, redact], outputs=[ingest_status, preview_text])

    with gr.Tab("2) Query"):
        with gr.Row():
            query = gr.Textbox(label="Your question", value="What is the sanctioned amount and EMI?")
            k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            use_llm = gr.Checkbox(value=False, label="Use LLM synthesis (set OPENAI_API_KEY in Space Secrets)")
        ask_btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=6)
        contexts = gr.Textbox(label="Retrieved Contexts", lines=12)
        scores = gr.Textbox(label="Scores", lines=2)
        fields = gr.JSON(label="Extracted Fields (SanctionedAmount, EMI, ROI, Tenure, etc.)")
        ask_btn.click(ask_query, inputs=[query, k, use_llm], outputs=[answer, contexts, scores, fields])

if __name__ == "__main__":
    demo.launch()
