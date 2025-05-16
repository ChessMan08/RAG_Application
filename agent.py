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

    # 3) RAG + LLM
    snippets = retrieve(query, k=3)
    context = "\n\n".join(snippets)

    prompt = f"""Answer the question below using only the information in the context.  
If the context doesn’t contain the answer, say “"I’m sorry, I don’t have enough information to answer that. Please try asking in a different way or about another topic.”

Context:
{context}

Question: {query}
Answer:"""

    gen = get_generator()
    out = gen(prompt, max_length=200, do_sample=False)[0]["generated_text"].strip()

    return {
        "branch":"rag",
        "snippets": snippets,
        "answer": out,
        "log": f"RAG+FlanT5 generated answer"
    }
