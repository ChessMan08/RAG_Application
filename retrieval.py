import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from ingestion import build_faiss_index, INDEX_FILE, CHUNKS_FILE

# Ensure index exists
if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    build_faiss_index(all_chunks)

# Load
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, k: int = 3):
    vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    _, ids = index.search(vec, k)
    return [chunks[i] for i in ids[0]]
