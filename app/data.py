# app/data.py
import json
from pathlib import Path
import requests

API_URL = API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
CACHE_DIR = Path("data_cache")
CACHE_MESSAGES = CACHE_DIR / "corpus_messages.json"



def fetch_first_page(limit: int = 100) -> list[dict]:
    """Fetch the first (and only usable) page of messages."""
    params = {"skip": 0, "limit": limit}
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()

    data = resp.json()
    return data.get("items", [])


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message schema for internal usage."""
    normalized = []
    for m in messages:
        normalized.append({
            "message_id": m.get("id"),
            "user_id": m.get("user_id"),
            "user_name": m.get("user_name"),
            "text": (m.get("message") or "").strip(),
            "timestamp": m.get("timestamp"),
        })
    return normalized


def load_corpus() -> list[dict]:
    """Load cached messages if available; otherwise fetch the first page."""
    CACHE_DIR.mkdir(exist_ok=True)

    # Load from cache
    if CACHE_MESSAGES.exists():
        print("[INFO] Loading messages from cache...")
        with open(CACHE_MESSAGES, "r") as f:
            return json.load(f)

    # Fetch fresh
    print("[INFO] Fetching first page of messages...")
    messages = fetch_first_page(limit=100)

    normalized = normalize_messages(messages)

    # Cache it
    with open(CACHE_MESSAGES, "w") as f:
        json.dump(normalized, f, indent=2)

    return normalized