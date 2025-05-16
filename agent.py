# agent.py

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
    q = query.lower().strip()

    # 1) Calculator branch
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {
            "branch": "calculator",
            "snippets": [],
            "answer": ans,
            "log": f"Calculated '{expr}' → {ans}"
        }

    # 2) Dictionary branch
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {
            "branch": "dictionary",
            "snippets": [],
            "answer": ans,
            "log": f"Defined '{term}' → {ans}"
        }

    # 3) RAG–semantic-match branch
    # retrieve top-3 chunks
    snippets = retrieve(query, k=3)

    best_answer = None
    best_snippet = None
    best_score = -1.0

    # Normalize query for matching
    q_norm = q.rstrip("?")

    # Iterate through each retrieved chunk
    for chunk in snippets:
        # Split chunk into segments at each "Q:"
        for seg in chunk.split("Q:"):
            text = seg.strip()
            if not text or "A:" not in text:
                continue
            ques_part, ans_part = text.split("A:", 1)
            ques_norm = ques_part.lower().strip().rstrip("?")
            # If the normalized user query is contained in this segment's question
            if q_norm in ques_norm:
                best_answer = ans_part.split("Q:", 1)[0].strip()
                best_snippet = chunk
                best_score = 1.0
                break
        if best_answer:
            break

    # Fallback if no segment matched
    if not best_answer:
        first = snippets[0]
        if "A:" in first:
            best_answer = first.split("A:", 1)[1].split("Q:", 1)[0].strip()
        else:
            best_answer = first.strip()
        best_snippet = first

    return {
        "branch": "rag",
        "snippets": [best_snippet],
        "answer": best_answer,
        "log": f"RAG semantic-matched snippet (score={best_score})"
    }
