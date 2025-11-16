# app/embeddings.py

import numpy as np
from pathlib import Path
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

CACHE_DIR = Path("data_cache")
CACHE_EMBEDDINGS = CACHE_DIR / "corpus_embeddings.npy"

def load_embeddings():
    """Load precomputed embeddings from disk."""
    if not CACHE_EMBEDDINGS.exists():
        raise RuntimeError(
            f"Embedding file not found: {CACHE_EMBEDDINGS}. "
            "You must compute embeddings locally before deploying."
        )

    return np.load(CACHE_EMBEDDINGS)


def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    embs = model.encode(texts, convert_to_numpy=True)

    # Ensure (1, d) shape for single query
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)

    return embs


# Render will call this on startup
def load_or_compute_embeddings(messages):
    """Load embeddings only. No computing on Render."""
    print("[INFO] Loading embeddings from cache...")
    return load_embeddings()