from app.data import fetch_first_page

messages = fetch_first_page(limit=5)

print("Fetched", len(messages), "messages.")
for msg in messages:
    print(msg)