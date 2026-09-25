import json
import urllib.request


URL = "http://127.0.0.1:8000/analyze"


TEST_CASES = [
    {
        "name": "Normal diary",
        "text": "今天下午上完课以后和朋友去食堂吃饭，然后回宿舍写作业。"
    },
    {
        "name": "Obvious promotion",
        "text": "这个精华真的太好用了，用了一周皮肤明显变亮，现在还有限时优惠，姐妹们赶紧冲！"
    },
    {
        "name": "Subtle recommendation",
        "text": "最近一直在用这个防晒，本来没抱太大希望，但是感觉确实没有以前那么容易晒黑。"
    },
    {
        "name": "Negative review",
        "text": "买了网上很火的这款面膜，用了几次感觉一般，补水还行，但是完全没有博主说得那么神。"
    },
    {
        "name": "Shopping share",
        "text": "最近买了几件秋天的衣服，整理一下分享给大家，有几件我自己还挺喜欢。"
    },
]


def analyze(text):
    payload = json.dumps(
        {"text": text},
        ensure_ascii=False
    ).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")

        return (
            response.status,
            json.loads(body)
        )


def main():
    print("=" * 70)
    print("LAYER 2 END-TO-END TEST")
    print("=" * 70)

    passed = 0

    for case in TEST_CASES:
        print("\n" + "-" * 70)
        print("TEST:", case["name"])
        print("TEXT:", case["text"])

        try:
            status, result = analyze(
                case["text"]
            )

            print("HTTP:", status)

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )
            )

            assert status == 200
            assert "authorship" in result
            assert "covert_ad" in result

            ad = result["covert_ad"]

            assert "label" in ad
            assert "probability" in ad
            assert "threshold" in ad

            probability = ad["probability"]

            assert 0 <= probability <= 1
            assert ad["threshold"] == 0.55

            print("PASS")
            passed += 1

        except Exception as e:
            print("FAIL")
            print("Error:", e)

    print("\n" + "=" * 70)
    print(f"Passed: {passed}/{len(TEST_CASES)}")

    if passed == len(TEST_CASES):
        print(
            "PASS: Layer 2 API integration works end-to-end."
        )
    else:
        print(
            "FAIL: Some Layer 2 integration tests failed."
        )


if __name__ == "__main__":
    main()