from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
