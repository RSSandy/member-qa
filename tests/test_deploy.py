import requests

url = "https://member-qa-6wh6.onrender.com/ask"

resp = requests.post(url, json={
    "question": "Where does Layla want a dinner reservation for her family?"
})

print(resp.status_code)
print("RAW TEXT:", resp.text)

