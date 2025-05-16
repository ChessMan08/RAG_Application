import os
import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG‑Powered Multi‑Agent Q&A",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("RAG‑Powered Multi‑Agent Q&A")

# Ensure FAISS index exists
from ingestion import build_faiss_index, INDEX_FILE, CHUNKS_FILE
if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        build_faiss_index()

# Import the agent
from agent import handle_query

# Session state for logs & history
if "logs" not in st.session_state:
    st.session_state.logs = []
if "history" not in st.session_state:
    st.session_state.history = [] 

with st.form("question_form", clear_on_submit=True):
    batch = st.text_area(
        "Ask Questions",
        height=68,
        key="batch_input"
    )
    submitted = st.form_submit_button("Submit")

if submitted and batch.strip():
    questions = [q.strip() for q in batch.splitlines() if q.strip()]
    for q in questions:
        res = handle_query(q)
        st.session_state.logs.append(f"Q: {q} | {res['log']}")
        st.session_state.history.append({
            "q": q,
            "branch": res["branch"],
            "snippets": res["snippets"],
            "answer": res["answer"]
        })


# Display all results
if st.session_state.history:
    st.markdown("---")
    st.header("Batch Results")
    for entry in st.session_state.history:
        st.subheader(f"Q: {entry['q']}")
        st.write("**Branch:**", entry["branch"].upper())
        if entry["branch"] == "rag" and entry["snippets"]:
            st.write("**Source Snippet:**")
            st.markdown(f"> {entry['snippets'][0]}")
        st.write("**Answer:**", entry["answer"])
        st.markdown("---")

# Sidebar: full agent log
with st.sidebar:
    st.header("Agent Log")
    for line in st.session_state.logs:
        st.write(line)
