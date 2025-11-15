# app/embeddings.py

import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = Path("data_cache")
CACHE_EMBEDDINGS = CACHE_DIR / "corpus_embeddings.npy"

# Load the embedding model ONCE at import time
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
#  Embedding helpers
# -------------------------
def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    SentenceTransformers sometimes returns shape (d,)
    when running inside forked pytest workers.
    
    This function guarantees shape is always (1, d).
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def embed_text(text: str) -> np.ndarray:
    emb = model.encode([text], convert_to_numpy=True)
    return _ensure_2d(emb)


def embed_texts(text_list: List[str]) -> np.ndarray:
    emb = model.encode(text_list, convert_to_numpy=True)
    return _ensure_2d(emb)


def embed_corpus(messages: List[dict]) -> np.ndarray:
    texts = [m["text"] for m in messages]
    emb = embed_texts(texts)
    return emb


# -------------------------
#  Caching
# -------------------------
def load_or_compute_embeddings(messages: List[dict]) -> np.ndarray:
    CACHE_DIR.mkdir(exist_ok=True)

    if CACHE_EMBEDDINGS.exists():
        print("[INFO] Loading embeddings from cache...")
        emb = np.load(CACHE_EMBEDDINGS)
        return _ensure_2d(emb)

    print("[INFO] Computing embeddings...")
    emb = embed_corpus(messages)
    np.save(CACHE_EMBEDDINGS, emb)
    return emb
