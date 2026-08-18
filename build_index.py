"""
Builds a retrieval index over the knowledge base.

Design choice: TF-IDF + FAISS (instead of a neural embedding model) so the
whole pipeline runs offline with no external model downloads. Swap in
sentence-transformers or an API embedding model later if you want denser
retrieval — the rest of the pipeline (agent.py, eval/*) doesn't need to
change, since it only calls `retrieve()`.
"""
import os
import pickle

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


def chunk_text(text, source, chunk_size=500, overlap=100):
    """Simple sliding-window chunker over raw characters, chunk boundaries
    snapped to paragraph breaks where possible. Consecutive chunks share
    up to `overlap` trailing/leading characters, so a detail sitting right
    at a chunk boundary doesn't get orphaned from the paragraph that gives
    it context."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
                # seed the next chunk with the tail of this one instead of
                # starting from a hard cut
                tail = buf[-overlap:] if overlap > 0 else ""
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return [{"text": c, "source": source} for c in chunks]


# Explicit whitelist rather than glob("*.md") -- the repo root also holds
# README.md, report.md, and failure_analysis.md, which are project docs,
# not Nimbus Notes knowledge-base content, and must NOT be indexed.
KB_FILES = [
    "account_management.md",
    "features.md",
    "pricing.md",
    "privacy_security.md",
    "troubleshooting.md",
]


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    all_chunks = []
    for filename in KB_FILES:
        path = os.path.join(KB_DIR, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        all_chunks.extend(chunk_text(text, source))

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts).toarray().astype("float32")

    # normalize so inner product == cosine similarity
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_matrix = tfidf_matrix / norms

    index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
    index.add(tfidf_matrix)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Indexed {len(all_chunks)} chunks from {len(set(c['source'] for c in all_chunks))} documents.")


if __name__ == "__main__":
    build_index()
