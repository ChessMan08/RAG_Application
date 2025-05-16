import os
import glob
import pickle
import nltk
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download WordNet once
nltk.download('wordnet')

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

def build_faiss_index(docs_path: str = "docs/"):
    # Load .txt files
    files = glob.glob(os.path.join(docs_path, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {docs_path}")
    texts = [open(f, encoding='utf-8').read() for f in files]

    # Chunk texts
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    # Embed
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype('float32')

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Built index with {len(chunks)} chunks.")

if __name__ == "__main__":
    build_faiss_index()
