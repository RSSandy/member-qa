from fastapi.testclient import TestClient
from app.main import app
import app.embeddings as emb
import csv
import pytest

# --- Load data after startup event ---
def get_loaded_state():
    with TestClient(app):  # triggers startup_event
        pass
    return app.state.corpus_messages, app.state.corpus_embeddings

corpus_messages, corpus_embeddings = get_loaded_state()



client = TestClient(app)


# -------------------------------
# Load CSV test cases
# -------------------------------
def load_qa_csv(path="tests/qa_answers.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"],
                "expected": row["expected"].lower().strip(),
            })
    return rows


qa_rows = load_qa_csv()


# -------------------------------
# Helper: Print retrieval debug info
# -------------------------------
def debug_retrieval(question, k=5):
    from sklearn.metrics.pairwise import cosine_similarity

    q_emb = emb.embed_texts(question)  # shape (1, dim)
    sims = cosine_similarity(q_emb, corpus_embeddings)[0]

    top_idx = sims.argsort()[::-1][:k]

    print("\n🔎 TOP RETRIEVED MESSAGES:")
    for rank, idx in enumerate(top_idx, start=1):
        m = corpus_messages[idx]
        print(f"\n  #{rank}  (score={sims[idx]:.4f})")
        print(f"    ID:   {m['message_id']}")
        print(f"    User: {m['user_name']}")
        print(f"    Text: {m['text']}")


# -------------------------------
# MAIN TEST
# -------------------------------
@pytest.mark.parametrize("row", qa_rows)
def test_expected_answers(row):
    question = row["question"]
    expected = row["expected"]

    print("\n=======================================================")
    print("QUESTION:", question)
    print("=======================================================")

    # ---- Debug retrieval BEFORE calling API ----
    debug_retrieval(question)

    # ---- Call API ----
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200

    answer = response.json().get("answer", "").lower()

    print("\n🟦 FINAL MODEL ANSWER:")
    print(answer)
    print()

    # ---- Assertion ----
    assert expected in answer, (
        f"Expected substring '{expected}' not found in answer: {answer}"
    )
