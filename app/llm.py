import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a question-answering system.
You must answer ONLY using the provided messages.
If the answer cannot be found in those messages, say "I don't know."
"""

def answer_with_llm(question: str, context_messages: list[str]) -> str:
    context = "\n".join(context_messages)

    prompt = f"""
    Context:
    {context}

    Question: {question}
    Answer based ONLY on the context above.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0,   # deterministic
    )

    return response.choices[0].message.content.strip()
