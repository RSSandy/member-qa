from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.data import load_corpus
from app.embeddings import (
    corpus_messages, 
    corpus_embeddings, 
    load_or_compute_embeddings,
)
from app.parsing import parse_question_llm
from app.retrieval import retrieve_relevant_messages
from app.answer import generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global corpus_messages, corpus_embeddings

    print("[INFO] Loading messages (cache-aware)...")
    corpus_messages = load_corpus()
    print(f"[INFO] Loaded {len(corpus_messages)} messages.")

    print("[INFO] Preparing embeddings (cache-aware)...")
    corpus_embeddings = load_or_compute_embeddings(corpus_messages)
    print("[INFO] Embeddings ready.")

    yield

    print("[INFO] App shutdown.")

app = FastAPI(lifespan=lifespan)

## Request Model
class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1) Parse the question
    parsed = parse_question_llm(question)

    # 2) Retrieve relevant messages
    retrieved = retrieve_relevant_messages(
        question=question,
        user_name=parsed.get("user_name"),
        k=5
    )

    # 3) Generate the final answer
    answer = generate_answer(
        question=question,
        parsed=parsed,
        retrieved_messages=retrieved
    )

    return answer