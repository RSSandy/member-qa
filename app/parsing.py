# app/parsing.py

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("[INFO] Loading local LLM for parsing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def parse_question_llm(question: str) -> dict:
    """
    Uses a small local LLM to extract:
    - user_name (best guess)
    - intent (cars, travel plans, restaurants, etc.)
    - entities (like 'London')
    """

    prompt = f"""
You are an information extraction assistant. 
Extract structured data from the user's question. 
Return **ONLY valid JSON** with the following fields:

- "user_name": The name of the member mentioned in the question. If none, set to null.
- "intent": What the user wants to know (e.g., "travel_plans", "car_count", "restaurant_preferences").
- "entities": A list of important places, dates, or nouns.
- "raw": The original question.

Question: "{question}"

Return JSON only. No explanation.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,  # deterministic
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract last JSON object from the output
    try:
        # Find first "{" and last "}" to isolate JSON
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except:
        return {
            "user_name": None,
            "intent": None,
            "entities": [],
            "raw": question,
            "error": "Could not parse model output"
        }
