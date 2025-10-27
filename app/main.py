import os, uuid
from fastapi import FastAPI, UploadFile, File
from .types import QueryRequest, IngestResponse, QueryResponse
from .ocr_utils import ocr_file
from .rag import RAGPipeline

STORE_DIR = os.getenv("STORE_DIR", os.path.join(os.path.dirname(__file__), "..", "store"))
STORE_DIR = os.path.abspath(STORE_DIR)

app = FastAPI(title="RAG OCR Demo", version="0.1.0")
pipeline = RAGPipeline(STORE_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), lang: str = "eng", redact_pii: bool = True):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    raw = await file.read()
    path = os.path.join(data_dir, file.filename)
    with open(path, "wb") as f:
        f.write(raw)
    text = ocr_file(path, lang=lang)
    doc_id = f"{file.filename}-{uuid.uuid4().hex[:8]}"
    chunks, tokens = pipeline.ingest_text(text, doc_id=doc_id, redact=redact_pii)
    return IngestResponse(doc_id=doc_id, chunks=chunks, tokens=tokens)

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    answer, contexts, scores = pipeline.answer(req.query, k=req.k, use_llm=req.use_llm)
    return QueryResponse(answer=answer, contexts=contexts, scores=scores)
