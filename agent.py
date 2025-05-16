from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

from transformers import pipeline 

from sentence_transformers import SentenceTransformer
import numpy as np

LLM_MODEL = "google/flan-t5-small" 

_generator = None
def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            device=-1,        # CPU
            framework="pt"
        )
    return _generator

def handle_query(query: str) -> dict:
    """
    Routes query to:
      - calculator (if “calculate” in query)
      - dictionary (if “define” in query)
      - otherwise RAG → Flan‑T5 generation
    
    Returns dict with:
      branch: which tool
      snippets: retrieved chunks (for RAG)
      answer: generated or computed answer
      log: short description
    """
    q = query.lower().strip()

     # 1) Calculator
    if "calculate" in q:
        expr = q.replace("calculate","").strip()
        ans = calculate(expr)
        return {"branch":"calculator","snippets":[],"answer":ans,"log":f"Calculated '{expr}' → {ans}"}

    # 2) Dictionary
    if "define" in q:
        term = q.replace("define","").strip()
        ans = define(term)
        return {"branch":"dictionary","snippets":[],"answer":ans,"log":f"Defined '{term}' → {ans}"}

    # RAG–semantic-match branch
    snippets = retrieve(query, k=3)

    # We’ll pick the snippet whose “Q: …” most closely matches the user’s question text
    best_answer = None
    best_score = -1.0

    # Lower-case query for matching
    q_norm = query.lower().strip().rstrip('?')

    for chunk in snippets:
        # Split into segments at each "Q:" marker
        for seg in chunk.split("Q:"):
            text = seg.strip()
            if not text or "A:" not in text:
                continue
            ques_part, ans_part = text.split("A:", 1)
            # Normalize the question part
            ques_norm = ques_part.lower().strip().rstrip('?')
            # Simple substring match; you can enhance this to fuzzy matching
            if q_norm in ques_norm:
                # Found the matching Q/A pair
                best_answer = ans_part.split("Q:",1)[0].strip()
                best_score = 1.0
                break
        if best_answer:
            break

    # Fallback: if no snippet matched, fall back to the very first chunk’s answer
    if not best_answer:
        first = snippets[0]
        if "A:" in first:
            best_answer = first.split("A:",1)[1].split("Q:",1)[0].strip()
        else:
            best_answer = first.strip()

    return {
        "branch": "rag",
        "snippets": [chunk] if best_answer else [],
        "answer": best_answer,
        "log": f"RAG semantic-matched snippet → score={best_score}"
    }
