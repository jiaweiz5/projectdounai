from datasets import load_dataset


# ========================================
# ANXFOREVER
# ========================================

print("=" * 60)
print("ANXFOREVER")
print("=" * 60)

anx = load_dataset(
    "AnxForever/chinese-ai-detection-dataset"
)

print("Splits:", anx.keys())

print()
print("Example:")

print(anx["train"][0])


# ========================================
# CHASM
# ========================================

print()
print("=" * 60)
print("CHASM")
print("=" * 60)

chasm = load_dataset(
    "Jingyi77/CHASM-Covert_Advertisement_on_RedNote",
    split="Example",
    streaming=True,
)

chasm = chasm.select_columns([
    "id",
    "title",
    "description",
    "comments",
    "image_count",
    "label",
    "split",
])

example = next(iter(chasm))

print("CHASM fields:")
print(example.keys())