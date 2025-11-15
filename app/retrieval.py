# app/retrieval.py

from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity

from app.embeddings import (
    embed_text,
    corpus_messages,
    corpus_embeddings,
)


def retrieve_relevant_messages(
    question: str,
    user_name: Optional[str] = None,
    k: int = 5
) -> List[dict]:
    """
    Retrieve top-k messages relevant to the question.

    Steps:
    1. Embed the *raw question text* with SBERT
    2. Compute cosine similarities to all corpus embeddings
    3. Sort messages by similarity score
    4. If user_name detected → apply metadata filtering
    5. Return top-k messages
    """

    # Step 1: embed question
    q_emb = embed_text(question)  # shape (1, dim)

    # Step 2: cosine similarity across all messages
    sims = cosine_similarity(q_emb, corpus_embeddings)[0]  # shape (num_messages,)

    # Step 3: rank messages
    ranked_indices = sims.argsort()[::-1]  # best first

    ranked_messages = [corpus_messages[i] for i in ranked_indices]

    # Step 4: user_name filtering (optional)
    if user_name:
        user_name = user_name.lower()
        ranked_messages = [
            m for m in ranked_messages
            if m["user_name"] and user_name in m["user_name"].lower()
        ]

    # Step 5: return top-k
    return ranked_messages[:k]
