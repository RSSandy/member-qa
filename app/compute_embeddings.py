# compute_embeddings.py

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG — MUST MATCH YOUR APP
# -----------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim model
CACHE_DIR = Path("data_cache")
CACHE_EMBEDDINGS = CACHE_DIR / "corpus_embeddings.npy"
CACHE_MESSAGES = CACHE_DIR / "corpus_messages.json"


# -----------------------------
# STEP 1 — Load corpus messages
# -----------------------------
print(f"[INFO] Loading messages from {CACHE_MESSAGES}...")

if not CACHE_MESSAGES.exists():
    raise FileNotFoundError(
        f"Could not find {CACHE_MESSAGES}. Did you export your corpus?"
    )

with open(CACHE_MESSAGES, "r", encoding="utf-8") as f:
    messages = json.load(f)

# Expect: list of objects each with at least { "text": "..." }


# -----------------------------
# Extract texts
# -----------------------------
texts = [m["text"] for m in messages]

print(f"[INFO] Loaded {len(texts)} messages for embedding.")


# -----------------------------
# STEP 2 — Load model
# -----------------------------
print(f"[INFO] Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# STEP 3 — Compute embeddings
# -----------------------------
print("[INFO] Computing embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

print(f"[INFO] Finished embedding! Shape: {embeddings.shape}")  # Expect (N, 768)


# -----------------------------
# STEP 4 — Save embeddings
# -----------------------------
CACHE_DIR.mkdir(exist_ok=True)

np.save(CACHE_EMBEDDINGS, embeddings)
print(f"[INFO] Saved new embeddings → {CACHE_EMBEDDINGS}")

print("\n[SUCCESS] Local embedding rebuild complete!")
print("🎉 Now commit + push your updated corpus_embeddings.npy and deploy to Render.")
