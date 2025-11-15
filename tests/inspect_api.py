# tests/inspect_api.py

import requests
import time

API = "https://november7-730026606190.europe-west1.run.app/messages"

def test_basic():
    print("\n=== Test 1: Basic request ===")
    r = requests.get(API)
    print("Status:", r.status_code)
    print("Keys:", r.json().keys())
    print("Total:", r.json().get("total"))

def test_page(skip, limit=100):
    print(f"\n=== Test 2: skip={skip}, limit={limit} ===")
    try:
        r = requests.get(API, params={"skip": skip, "limit": limit})
        print("Status:", r.status_code)
        if r.status_code == 200:
            j = r.json()
            print("Items returned:", len(j.get("items", [])))
            return j
        else:
            print("Response text:", r.text[:250])
    except Exception as e:
        print("Exception:", e)

def test_sequential_paging():
    print("\n=== Test 3: Sequential paging 0,100,200,... ===")

    # First page: get total
    r0 = requests.get(API, params={"skip": 0, "limit": 100})
    if r0.status_code != 200:
        print("First page FAIL:", r0.status_code, r0.text)
        return
    total = r0.json()["total"]
    print("Total messages reported:", total)

    skip = 0
    while True:
        print(f"\n--- Requesting skip={skip} ---")
        r = requests.get(API, params={"skip": skip, "limit": 100})
        print("Status:", r.status_code)

        if r.status_code != 200:
            print("Error body:", r.text[:300])
            break

        data = r.json()
        num = len(data["items"])
        print("Items returned:", num)

        # Stop if server gives fewer items than limit
        if num < 100:
            print("Reached final partial page.")
            break

        skip += 100

        # Sleep to test throttling
        time.sleep(0.3)


def test_large_skips():
    print("\n=== Test 4: Large skip probes ===")

    probes = [500, 800, 1200, 1600, 2000, 5000]
    for sk in probes:
        test_page(sk)


def test_throttling():
    print("\n=== Test 5: Rapid-fire requests to test rate limiting ===")

    for i in range(10):
        print(f"\nRequest {i+1}/10")
        r = requests.get(API)
        print("Status:", r.status_code)
        if r.status_code != 200:
            print("Body:", r.text[:200])
        time.sleep(0.1)

def test_param_variations():
    print("\n=== Test 6: Parameter variation (what filters exist?) ===")

    tests = [
        {"user_name": "Layla"},
        {"userId": "1"},
        {"search": "London"},
        {"text": "London"},
        {"username": "Hans"},
    ]

    for params in tests:
        print(f"\nParams: {params}")
        try:
            r = requests.get(API, params=params)
            print("Status:", r.status_code)
            if r.status_code == 200:
                print("Keys:", r.json().keys())
                print("Items:", len(r.json()["items"]))
            else:
                print("Body:", r.text[:200])
        except Exception as e:
            print("Exception:", e)


if __name__ == "__main__":
    test_basic()
    test_page(0)
    test_page(100)
    test_page(300)
    test_page(400)
    test_page(700)
    test_page(1600)

    test_sequential_paging()
    test_large_skips()
    test_throttling()
    test_param_variations()
