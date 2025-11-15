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

# Global variables to be filled at startup
corpus_messages: List[dict] = []
corpus_embeddings: np.ndarray = None


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string into a vector."""
    return model.encode([text], convert_to_numpy=True)


def embed_texts(text_list: List[str]) -> np.ndarray:
    """Embed multiple texts into a matrix."""
    return model.encode(text_list, convert_to_numpy=True)


def embed_corpus(messages: List[dict]) -> np.ndarray:
    """Embed all message texts from the corpus."""
    texts = [m["text"] for m in messages]
    return embed_texts(texts)

def load_or_compute_embeddings(messages: List[dict]) -> np.ndarray:
    """Load embeddings from cache if present; compute and save otherwise."""
    CACHE_DIR.mkdir(exist_ok=True)

    if CACHE_EMBEDDINGS.exists():
        print("[INFO] Loading embeddings from cache...")
        return np.load(CACHE_EMBEDDINGS)

    print("[INFO] Computing embeddings...")
    emb = embed_corpus(messages)
    np.save(CACHE_EMBEDDINGS, emb)
    return emb

def retrieve_relevant_messages(question: str, k: int = 5) -> List[dict]:
    """Return top-k most relevant messages to the given question."""
    global corpus_messages, corpus_embeddings

    if corpus_embeddings is None or len(corpus_messages) == 0:
        raise RuntimeError("Corpus not initialized. Did you run startup_event()?")

    # Embed question
    q_emb = embed_text(question)  # shape (1, dim)

    # Cosine similarity to all corpus embeddings
    sims = cosine_similarity(q_emb, corpus_embeddings)[0]  # shape (num_messages,)

    # Pick top-k highest scoring indices
    top_indices = sims.argsort()[::-1][:k]

    # Return messages in ranked order
    return [corpus_messages[i] for i in top_indices]
