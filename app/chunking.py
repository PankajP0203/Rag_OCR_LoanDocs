import re
from typing import List

def clean_text(text: str) -> str:
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def simple_paragraph_split(text: str) -> List[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def token_estimate(s: str) -> int:
    return max(1, len(s) // 4)

def chunk_paragraphs(paras: List[str], max_tokens: int = 350, overlap_tokens: int = 40) -> List[str]:
    chunks, cur, cur_tokens = [], [], 0
    for p in paras:
        t = token_estimate(p)
        if cur_tokens + t <= max_tokens:
            cur.append(p); cur_tokens += t
        else:
            if cur: chunks.append('\n'.join(cur))
            if chunks and overlap_tokens > 0:
                tail = chunks[-1].split()[-overlap_tokens:]
                cur = [' '.join(tail), p]
                cur_tokens = token_estimate(cur[0]) + t
            else:
                cur = [p]; cur_tokens = t
    if cur: chunks.append('\n'.join(cur))
    return chunks
