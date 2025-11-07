import os, re
from .retriever import Retriever
from .chunking import clean_text, simple_paragraph_split, chunk_paragraphs, token_estimate

def redact_pii(text: str) -> str:
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', 'PAN_[REDACTED]', text)
    text = re.sub(r'\b\d{12}\b', 'AADHAAR_[REDACTED]', text)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', 'IFSC_[REDACTED]', text)
    text = re.sub(r'\b\d{9,18}\b', 'ACNO_[REDACTED]', text)
    return text

# app/rag.py
def synthesize_answer(query: str, contexts: list[str]) -> str:
    import os
    combined = "\n\n".join([c for c in contexts if c])[:12000]  # trim long context
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    if not api_key:
        return "[LLM not configured]\n\n" + (combined[:1500] if combined else "No context available.")

    try:
        from openai import OpenAI
        import httpx  # used only if proxy is set

        # If you *really* need a proxy, set HTTPS_PROXY / HTTP_PROXY in Space secrets.
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
        if proxy:
            http_client = httpx.Client(proxies=proxy, timeout=30.0)
            client = OpenAI(api_key=api_key, http_client=http_client)
        else:
            client = OpenAI(api_key=api_key)

        system = (
            "You are a precise loan document assistant. "
            "Answer ONLY using the CONTEXT. If information is missing, say so. "
            "Prefer concise numeric answers for sanctioned amount, EMI, ROI, tenure."
        )
        prompt = f"CONTEXT:\n{combined}\n\nQUESTION: {query}\n\nAnswer:"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error] {e}\n\n{combined[:1500]}"
class RAGPipeline:
    def __init__(self, store_dir: str):
        self.retriever = Retriever(store_dir)

    def ingest_text(self, raw_text: str, doc_id: str, redact: bool = True):
        text = clean_text(raw_text or "")
        if redact:
            text = redact_pii(text)
        if not text.strip():
            # nothing to ingest
            return 0, 0
        paras = simple_paragraph_split(text)
        chunks = chunk_paragraphs(paras, max_tokens=350, overlap_tokens=40)
        if not chunks:
            return 0, 0
        self.retriever.add_documents(chunks, doc_id=doc_id)
        tokens = sum(token_estimate(c) for c in chunks)
        return len(chunks), tokens

    def answer(self, query: str, k: int = 5, use_llm: bool = True):
        scores, texts = self.retriever.query(query, k=k)
        ans = synthesize_answer(query, texts) if use_llm else "\n\n".join(texts[:3])
        return ans, texts, scores
