---
title: "Member QA"
emoji: "🔍"
colorFrom: "yellow"
colorTo: "purple"
sdk: "docker"
app_file: "app.py"
pinned: false
---

# Member QA — Natural-Language Question Answering API
Built by Sandhya Nayar

🌐 Live API: https://rs-snayar-member-qa.hf.space

📄 Interactive Docs: https://rs-snayar-member-qa.hf.space/

Demo: https://www.loom.com/share/b9055e77d8e64aa990c2b88bed0e3323

## 🚀 Overview

This project implements a deployed question-answering API that accepts natural-language questions about member activity and returns an answer inferred from a set of historical member messages.

The service is:
- Fully deployed and publicly accessible (Hugging Face Space)
- Implemented with FastAPI
- Backed by semantic search using precomputed embeddings
- Capable of handling flexible natural-language questions
- Able to recover answers even from ambiguous or indirect queries

The system uses:

- A custom RAG-style retrieval pipeline
- Robust question parsing with an LLM
- Embedding-based similarity search
- A deterministic answer selection layer

The result is a lightweight, production-ready API that consistently returns JSON:

```json
{ "answer": "string" }
```

System Architecture

```CSS
 ┌─────────────────────────┐
 │ Incoming User Question  │
 └─────────────┬───────────┘
               │
               ▼
   ┌─────────────────────────────┐
   │ 1. LLM-based Question Parser│
   │    • extracts entities      │
   │    • canonicalizes names    │
   │    • identifies the intent  │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 2. Semantic Retriever       │
   │    • precomputed embeddings │
   │    • cosine similarity      │
   │    • top-k message ranking  │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 3. Answer Synthesizer       │
   │    • extracts key fields    │
   │    • formats final answer   │
   │    • returns JSON only      │
   └─────────────────────────────┘
```

## 🔍 Alternative Approaches Considered (and Why They Were Rejected)

During development, I explored multiple possible approaches.
Below is a summary of what I tried (or considered) and why each option was rejected.

#### ❌ 1. Using Render.com (Initially Attempted)

Render had poor Python library support, including:
- ML packages missing
- Slow cold starts
- Missing dependencies (sentence-transformers, sklearn)
- Import failures for OpenAI and tokenizer libraries

After repeated failures to install core dependencies, I switched to Hugging Face Spaces, which have far better ML and Python support.

#### ❌ 2. Using only rule-based NLP + regex

I considered writing a pure heuristic system:
- entity extractors
- rule matching
- string search patterns

But this quickly failed:

- Couldn’t handle ambiguous phrasing
- Totally broke on pronouns (“she”, “they”, “their booking”)
- Fails on paraphrasing (“need seats for” vs “reserve seats for”)
- Very brittle for real language

This approach was incompatible with natural-language questions.

#### ❌ 3. Letting an LLM directly generate the JSON answer

I tried variants where:
- an LLM read the question
- read the messages
- generated the final {"answer": ...} JSON directly

This was fragile:

- LLMs often hallucinate
- JSON validity is unreliable without special constraints
- Hard to enforce determinism
- Needed a very capable (expensive) model to ensure correctness

### ❌ 4. Letting the LLM perform retrieval itself

This was rejected because:

- retrieval must be deterministic
- LLMs vary across calls
- embedding-based search is much more predictable
- assignment requires inference from the provided messages

We want the LLM to parse the question, not the data.


## 📊 Bonus: Data Insights / Anomalies

I examined the messages for anomalies, didn't notice any literal data inconsistencies (format errors, corrupted text, etc.) were found.

However, an interesting observation surfaced: Multiple messages can legitimately answer a single open-ended question.

Example: “Where did Layla want a dinner reservation?”

Potential answers:
One message may refer to a reservation in Florida
Another may refer to a different date or different restaurant

Both are correct in different contexts

This ambiguity means:

- Retrieval must consider top K, not just the nearest neighbor
- Answer synthesis must choose based on context
- Some questions do not have a unique correct answer

This is a natural limitation of real-world conversational data.
