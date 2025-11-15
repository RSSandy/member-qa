import json
import os
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_context(messages: List[dict]) -> str:
    if not messages:
        return "NO_RELEVANT_MESSAGES"

    lines = []
    for m in messages:
        username = m.get("user") or m.get("user_name") or "UNKNOWN"
        text = json.dumps(m.get("text", ""))
        lines.append(f"- {username}: {text}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    parsed: dict,
    retrieved_messages: List[dict]
) -> Dict:

    context_block = format_context(retrieved_messages)

    prompt = f"""
You are an assistant that answers questions ONLY using the provided user messages.

If the information is not contained in the messages, respond exactly with:
Sorry, I couldn't find that information.

Otherwise respond with ONLY the answer, in SHORT plain text.
No JSON. No extra words. No quotes. No punctuation unless part of the answer.

QUESTION:
{question}

RETRIEVED_MESSAGES:
{context_block}

ANSWER:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Respond ONLY with the answer text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.0
    )

    answer_text = response.choices[0].message.content.strip()

    # wrap it in JSON yourself
    return {"answer": answer_text}
