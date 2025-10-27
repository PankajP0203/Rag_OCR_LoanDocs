import os, json, faiss, numpy as np
from typing import List, Dict

class FaissStore:
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.index_path = os.path.join(self.db_dir, "index.faiss")
        self.meta_path = os.path.join(self.db_dir, "meta.json")
        self.index = None
        self.metadata: List[Dict] = []

    def _load(self, dim: int):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        dim = embeddings.shape[1]
        if self.index is None:
            self._load(dim)
        elif self.index.d != dim:
            raise ValueError(f"Embedding dim mismatch: {self.index.d} vs {dim}")
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)
        self._persist()

    def search(self, query_emb: np.ndarray, k: int = 5):
        if self.index is None:
            raise ValueError("Index not loaded")
        D, I = self.index.search(query_emb.astype("float32"), k)
        texts = [self.metadata[i] if i != -1 else None for i in I[0]]
        return D[0], texts

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, dim: int):
        self._load(dim)
