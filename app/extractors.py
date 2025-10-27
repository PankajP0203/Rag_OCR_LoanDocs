# app/extractors.py
import re
from typing import Dict

_PATTERNS = [
    ("SanctionedAmount", r"(?:Loan Amount Sanctioned.*?:\s*[_\|\s]*)(₹?\s?[\d,]+)"),
    ("EMI",               r"(?:Amount of EMI.*?:|EMI\s*[:\-])\s*₹?\s?([\d,]+)"),
    ("ROI",               r"(?:Floating Interest Rate.*?[-–]\s*|ROI.*?:\s*)(\d+(?:\.\d+)?%?)"),
    ("Tenure",            r"(?:Loan Tenor.*?:|Tenure.*?:)\s*([\d]+\s*(?:years?|months?))"),
    ("SanctionDate",      r"(?:Sanction(?:ed)? Date.*?:\s*)([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})"),
]

def _norm(s: str) -> str:
    s = re.sub(r'[_\|]+', '', s)         # remove stray OCR artifacts
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def extract_fields(text: str) -> Dict[str, str]:
    text = _norm(text)
    found: Dict[str, str] = {}
    for key, pat in _PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        val = m.group(m.lastindex or 1)
        found[key] = _norm(val)
    return found
