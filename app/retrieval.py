# app/retrieval.py

from sklearn.metrics.pairwise import cosine_similarity
import app.embeddings as emb


def retrieve_relevant_messages(question, user_name, k, request=None):
    messages = request.app.state.corpus_messages
    embeddings = request.app.state.corpus_embeddings

    q_emb = emb.embed_texts(question)  # shape (1, dim)
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked_indices = sims.argsort()[::-1][:k]

    ranked_messages = [messages[i] for i in ranked_indices]

    if user_name:
        user_name = user_name.lower()

        filtered = [
            m for m in ranked_messages
            if user_name in m["user_name"].lower()
        ]

        if filtered:
            ranked_messages = filtered

    return ranked_messages