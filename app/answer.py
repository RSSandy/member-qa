# app/answer.py

import json
import os
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_context(messages: List[dict]) -> str:
    """
    Convert retrieved messages into a readable block.
    """
    if not messages:
        return "NO_RELEVANT_MESSAGES"

    lines = []
    for m in messages:
        lines.append(
            f"- {m['user_name']}: {m['text']}  (timestamp: {m['timestamp']})"
        )
    return "\n".join(lines)


def generate_answer(
    question: str,
    parsed: dict,
    retrieved_messages: List[dict]
) -> Dict:
    """
    Generate final structured JSON answer using OpenAI.
    """

    context_block = format_context(retrieved_messages)

    prompt = f"""
You are an assistant that answers questions ONLY using the provided user messages.

If the information is not contained in the messages, respond with:
"Sorry, I couldn't find that information."

Return an answer in STRICT JSON form ONLY:
{{
  "answer": "..."
}}

QUESTION:
{question}

PARSED_DATA:
{json.dumps(parsed, indent=2)}

RETRIEVED_MESSAGES:
{context_block}

Now provide ONLY the JSON. No explanation, no extra text.
"""

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",        # cheap + good
        messages=[
            {"role": "system", "content": "You return ONLY valid JSON answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )

    text = response.choices[0].message.content.strip()

    # Extract JSON
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception:
        return {"answer": "Sorry, I couldn't generate a valid JSON answer."}
