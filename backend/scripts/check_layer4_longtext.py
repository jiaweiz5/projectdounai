"""Smoke-check the real SigLIP2 path on short and long post text."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_features import calculate_image_text_similarity


def main() -> None:
    """Use an existing development image to verify token windowing end to end."""
    dataset_path = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"
    with dataset_path.open(encoding="utf-8") as file:
        cases = json.load(file)
    case = next(
        item for item in cases
        if item["issue_type"] in {"none", "image_text_mismatch"}
    )
    image_path = BASE_DIR / case["image_path"]
    caption = case["post_text"]
    # Repetition here checks capacity and coverage, not classification accuracy.
    long_caption = (caption + " 补充说明。") * 6
    very_long_caption = (caption + " 补充说明。") * 30

    for name, text in (
        ("short", caption), ("long", long_caption),
        ("very long", very_long_caption),
    ):
        result = calculate_image_text_similarity(image_path, text)
        print(
            f"{name}: score={result['image_text_similarity']:.4f}, "
            f"windows={result['text_window_count']}, "
            f"covered={result['text_tokens_covered']}/"
            f"{result['text_token_count']} tokens"
        )
        if name == "short" and result["text_window_count"] != 1:
            raise AssertionError("Short development caption changed scoring paths.")
        if name == "long" and result["text_window_count"] <= 1:
            raise AssertionError("Long caption did not use text windows.")
        if name == "long" and result["text_coverage_ratio"] < 1.0:
            raise AssertionError("The moderately long caption was not fully covered.")
    print("PASS: both captions reached the image-text model without a 64-token error.")


if __name__ == "__main__":
    main()
