# app/parsing_openai.py
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_question(question: str) -> dict:
    """
    Use OpenAI GPT-4o-mini to extract:
    - user_name
    - intent
    - entities
    - raw question
    """

    prompt = f"""
Extract structured data from the user's question. Return ONLY valid JSON.

Fields:
- user_name: The person's name in the question (or null)
- intent: travel_plans, car_count, restaurant_preferences, etc.
- entities: list of important nouns
- raw: original question

Question: "{question}"

Return ONLY the JSON. No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Respond ONLY with strict JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return {
            "user_name": None,
            "intent": None,
            "entities": [],
            "raw": question
        }
