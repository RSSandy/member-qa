# main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.data import load_corpus
from app.embeddings import load_or_compute_embeddings
from app.retrieval import retrieve_relevant_messages

# ALWAYS use OpenAI parsing + answer generation
from app.parsing import parse_question
from app.answer import generate_answer


# -----------------------------
#   RATE LIMITER
# -----------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/minute"]
)

DEBUG_LAST = {
    "parsed": None,
    "retrieved": None
}


# -----------------------------
#   LIFESPAN
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Loading messages...")
    messages = load_corpus()

    print("[INFO] Computing embeddings...")
    embeddings = load_or_compute_embeddings(messages)

    app.state.corpus_messages = messages
    app.state.corpus_embeddings = embeddings

    yield



app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."}
    )


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
@limiter.limit("30/minute")
def ask(req: AskRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Parse the question with OpenAI
    parsed = parse_question(question)

    # Retrieve relevant messages
    retrieved = retrieve_relevant_messages(
        question=question,
        user_name=parsed.get("user_name"),
        k=5,
        request=request
    )

    # SAVE DEBUG INFO
    DEBUG_LAST["parsed"] = parsed
    DEBUG_LAST["retrieved"] = retrieved
    print("DEBUG PARSED:", parsed)
    
    # Generate final answer with OpenAI
    print("DEBUG retrieved passed into answer:", retrieved)
    answer = generate_answer(
        question=question,
        parsed=parsed,
        retrieved_messages=retrieved
    )

    return answer

@app.get("/debug/last")
def debug_last():
    return DEBUG_LAST
