import requests
import csv

# ---------------------------------------------------
# CONFIG: UPDATE THIS TO YOUR HUGGING FACE ENDPOINT
# ---------------------------------------------------
URL = "https://rs-snayar-member-qa.hf.space/ask"
# or your render.com endpoint:
# URL = "https://member-qa-6wh6.onrender.com/ask"
# ---------------------------------------------------


def load_qa_csv(path="tests/qa_answers.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"],
                "expected": row["expected"].lower().strip(),
            })
    return rows


def run_test(question, expected):
    print("\n=====================================================")
    print(f"QUESTION: {question}")
    print("=====================================================")

    try:
        resp = requests.post(URL, json={"question": question}, timeout=20)
    except Exception as e:
        print("❌ REQUEST ERROR:", e)
        return False

    print("Status:", resp.status_code)
    print("RAW TEXT:", resp.text)

    if resp.status_code != 200:
        print("❌ FAIL: Non-200 status.")
        return False

    try:
        answer = resp.json().get("answer", "").lower()
    except Exception:
        print("❌ FAIL: Could not parse JSON.")
        return False

    print("\n🟦 MODEL ANSWER:", answer)

    if expected in answer:
        print("✅ PASS")
        return True
    else:
        print(f"❌ FAIL: Expected '{expected}' not found in answer.")
        return False


def main():
    rows = load_qa_csv()
    passed = 0

    for row in rows:
        ok = run_test(row["question"], row["expected"])
        if ok:
            passed += 1

    print("\n=====================================================")
    print(f"FINAL SCORE: {passed}/{len(rows)} passed")
    print("=====================================================")


if __name__ == "__main__":
    main()
