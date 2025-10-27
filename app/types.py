from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    max_context_tokens: int = 2000
    redact_pii: bool = True
    use_llm: bool = True

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    tokens: int

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    scores: List[float]
