from pathlib import Path
import json
import urllib.request


URL = "http://127.0.0.1:8000/analyze"


TEST_CASES = [
    {
        "name": "Normal diary",
        "text": "今天下午上完课以后和朋友去食堂吃饭，然后回宿舍写作业。",
        "expected_label": "likely_non_ad",
    },
    {
        "name": "Obvious promotion",
        "text": "这个精华真的太好用了，用了一周皮肤明显变亮，现在还有限时优惠，姐妹们赶紧冲！",
        # No verified covert-ad label; report the prediction without scoring it.
    },
    {
        "name": "Subtle recommendation",
        "text": "最近一直在用这个防晒，本来没抱太大希望，但是感觉确实没有以前那么容易晒黑。",
    },
    {
        "name": "Negative review",
        "text": "买了网上很火的这款面膜，用了几次感觉一般，补水还行，但是完全没有博主说得那么神。",
        "expected_label": "likely_non_ad",
    },
    {
        "name": "Shopping share",
        "text": "最近买了几件秋天的衣服，整理一下分享给大家，有几件我自己还挺喜欢。",
    },
]


def analyze(text):
    payload = json.dumps(
        {"text": text},
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)

test_file = (
    Path(__file__).resolve().parents[1]
    / "data/processed/chasm_test.jsonl"
)

with test_file.open(encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        if int(row["label"]) == 1 and row["text"].strip():
            TEST_CASES.append({
                "name": "CHASM held-out ad",
                "text": row["text"],
                "expected_label": "likely_ad",
            })
            break
    else:
        raise RuntimeError("No labeled ad found in CHASM test file")

def main():
    print("=" * 70)
    print("LAYER 2 END-TO-END TEST")
    print("=" * 70)

    api_passed = 0
    prediction_checked = 0
    prediction_passed = 0

    for case in TEST_CASES:
        print("\n" + "-" * 70)
        print("TEST:", case["name"])
        print("TEXT:", case["text"])

        try:
            status, result = analyze(case["text"])

            print("HTTP:", status)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            assert status == 200, f"Expected HTTP 200, got {status}"
            assert "authorship" in result, "Missing authorship result"
            assert "covert_ad" in result, "Missing covert_ad result"

            ad = result["covert_ad"]

            assert "label" in ad, "Missing ad label"
            assert "probability" in ad, "Missing ad probability"
            assert "threshold" in ad, "Missing ad threshold"

            probability = ad["probability"]
            assert isinstance(probability, (int, float)), (
                "Probability must be a number"
            )
            assert 0 <= probability <= 1, (
                "Probability must be between 0 and 1"
            )
            assert ad["threshold"] == 0.55, (
                f"Expected threshold 0.55, got {ad['threshold']}"
            )

            api_passed += 1
            print("API: PASS")

            expected = case.get("expected_label")

            if expected is None:
                print("PREDICTION: NOT SCORED (no verified label)")
            else:
                prediction_checked += 1

                if ad["label"] == expected:
                    prediction_passed += 1
                    print("PREDICTION: PASS")
                else:
                    print(
                        "PREDICTION: FAIL — "
                        f"expected {expected}, got {ad['label']} "
                        f"(probability={probability})"
                    )

        except Exception as error:
            print("API: FAIL")
            print("Error:", error)

    print("\n" + "=" * 70)
    print(f"API integration: {api_passed}/{len(TEST_CASES)} passed")
    print(
        f"Labeled predictions: "
        f"{prediction_passed}/{prediction_checked} passed"
    )

    if not any(
        case.get("expected_label") == "likely_ad"
        for case in TEST_CASES
    ):
        print(
            "INCOMPLETE: Add a verified labeled ad case "
            "to check positive detection."
        )

    if api_passed != len(TEST_CASES):
        raise SystemExit(1)

    


if __name__ == "__main__":
    main()