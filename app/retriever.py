from typing import List
from .embeddings import embed_texts
from .vectorstore import FaissStore

class Retriever:
    def __init__(self, store_dir: str):
        self.store = FaissStore(store_dir)

    def add_documents(self, texts: List[str], doc_id: str) -> int:
        embs = embed_texts(texts)
        metas = [{"doc_id": doc_id, "text": t} for t in texts]
        self.store.add(embs, metas); return len(texts)

    def query(self, query: str, k: int = 5):
        qemb = embed_texts([query])
        self.store.load(qemb.shape[1])
        scores, metas = self.store.search(qemb, k=k)
        texts = [m["text"] if m else "" for m in metas]
        return scores.tolist(), texts
