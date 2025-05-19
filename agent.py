import re
from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Route the query to calculator, dictionary, or RAG–semantic-match branch.
    Returns a dict with:
      - branch: which tool was used
      - snippets: list containing the exact snippet used
      - answer: the extracted answer
      - log: a short description of the action
    """
    q = query.lower().strip().rstrip("?")

    # Calculator branch
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {
            "branch": "calculator",
            "snippets": [],
            "answer": ans,
            "log": f"Calculated '{expr}' → {ans}"
        }

    # Dictionary branch
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {
            "branch": "dictionary",
            "snippets": [],
            "answer": ans,
            "log": f"Defined '{term}' → {ans}"
        }

    # retrieve top-3 chunks
    snippets = retrieve(query, k=3)

    best_answer = None
    best_snippet = None
    best_score = 0.0

    # compile words from query
    query_words = set(re.findall(r"\w+", q))

    for chunk in snippets:
        for seg in chunk.split("Q:"):
            if "A:" not in seg:
                continue
            ques_part, ans_part = seg.split("A:", 1)
            # tokenize question part
            words = set(re.findall(r"\w+", ques_part.lower()))
            if not words: 
                continue
            score = len(query_words & words) / len(query_words | words)
            if score > best_score:
                best_score = score
                best_answer = ans_part.split("Q:",1)[0].strip()
                best_snippet = chunk

    # fallback if no good match
    if best_score < 0.1 or not best_answer:
        first = snippets[0]
        if "A:" in first:
            best_answer = first.split("A:",1)[1].split("Q:",1)[0].strip()
        else:
            best_answer = first.strip()
        best_snippet = first

    return {
        "branch": "rag",
        "snippets": [best_snippet],
        "answer": best_answer,
        "log": f"RAG matched snippet (jaccard={best_score:.2f})"
    }
