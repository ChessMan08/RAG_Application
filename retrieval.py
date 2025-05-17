import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Paths
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

# Load index & chunks (no automatic build here!)
if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        "Index files missing—run ingestion or upload flow first."
    )

# Load from disk
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, k: int = 3) -> list[str]:
    q_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    _, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0]]
