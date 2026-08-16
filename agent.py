"""
The RAG agent under test: retrieves relevant chunks with the TF-IDF/FAISS
index, then asks Gemini to answer using only the retrieved context.
"""
import os
import pickle
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

_index = None
_vectorizer = None
_chunks = None


def _load():
    global _index, _vectorizer, _chunks
    if _index is None:
        _index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "rb") as f:
            _vectorizer = pickle.load(f)
        with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
            _chunks = pickle.load(f)


def retrieve(query, k=3):
    _load()
    q_vec = _vectorizer.transform([query]).toarray().astype("float32")
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    scores, idxs = _index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({**_chunks[idx], "score": float(score)})
    return results


SYSTEM_PROMPT = """You are a support assistant for Nimbus Notes, a note-taking app.
Answer the user's question using ONLY the context provided below. If the answer is
not contained in the context, say clearly that you don't have that information and
do not guess. Be concise (2-4 sentences).

Context:
{context}
"""


def answer(question, k=3, model="gemini-3.5-flash-lite", client=None):
    """Runs one RAG query end-to-end. Returns dict with answer, latency,
    token usage, and the retrieved chunks (for failure analysis).

    `client` is a google.genai.Client() instance, passed in so callers
    (eval scripts) can reuse one client and so this module has no hard
    dependency on an API key being present at import time.
    """
    retrieved = retrieve(question, k=k)
    context = "\n\n---\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=question,
        config={
            "system_instruction": SYSTEM_PROMPT.format(context=context),
            "max_output_tokens": 300,
        },
    )
    latency = time.time() - start

    answer_text = response.text or ""
    usage = response.usage_metadata

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_sources": [r["source"] for r in retrieved],
        "retrieved_chunks": retrieved,
        "latency_seconds": latency,
        "input_tokens": usage.prompt_token_count if usage else 0,
        "output_tokens": usage.candidates_token_count if usage else 0,
    }
