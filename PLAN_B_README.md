# Layer 4 long-post patch

## Install in your existing repository

Copy the five files from this archive into the matching paths under
`/Users/jiaweizheng1/xhs-verifier/`. Review `git diff` before committing.

| File in archive | Purpose |
| --- | --- |
| `backend/layer4_features.py` | Token windows and SigLIP2 comparisons. |
| `backend/layer4_detector.py` | Marks uncalibrated long-post results provisional. |
| `backend/main.py` | Keeps other layers available when an image check fails. |
| `backend/scripts/calibrate_layer4.py` | Keeps the original short cutoff; optionally calibrates long posts separately. |
| `backend/scripts/check_layer4_longtext.py` | Runs short and long text through the real model. |

## Check locally

From `backend`, with your virtual environment active:

```bash
python -m py_compile main.py layer4_features.py layer4_detector.py scripts/calibrate_layer4.py scripts/check_layer4_longtext.py
python scripts/check_layer4_longtext.py
```

The smoke check should print one window for a short case and multiple windows
for both a moderately long and a very long caption, then `PASS`. This revision
uses the tokenizer and processor to add special tokens for each long window;
it does not call `GemmaTokenizer.build_inputs_with_special_tokens`, which is
missing in the tokenizer shipped with this model. It uses an image already referenced by
`data/layer4/layer4_dataset.json`; keep that image available on disk.

Run the API and submit a long caption with an image. The main request should
return HTTP 200 with Layers 1–3 intact and a `layer4.status` of `provisional`
until long-post calibration exists. Inspect
`layer4.images[0].alignment.text_coverage_ratio` for how much of the post was
compared. The optional Qwen report can still say `unavailable` if the Inference
Provider has no credits.

## Calibrate before treating long-post labels as reliable

The existing 60 cases are short captions. Running
`python scripts/calibrate_layer4.py` checks and regenerates that short-caption
cutoff. It does not validate the new highest-window score for long posts.

To calibrate long posts, create
`backend/data/layer4/layer4_long_alignment_cases.json` with **reviewed**
image/caption pairs from development data: at least five aligned (`label: 0`)
and five mismatched (`label: 1`). Each caption must need more than one text
window, and every case must have full text coverage. Use a separate unseen set
later to evaluate the chosen threshold. Each entry uses this shape:

```json
{
  "id": "unique_case_id",
  "image_path": "data/layer4/images/an_existing_image.jpg",
  "post_text": "A real long caption that has been reviewed against this image.",
  "label": 0
}
```

Wrap entries in a JSON array. Then run the calibration script again and
restart the API to clear its cached configuration. If the long file does not
exist, the code still returns a provisional result; it does not pretend that
the short-caption threshold was validated for long posts.

## Capacity and interpretation

SigLIP2 sees at most 64 tokens at a time. The patch compares at most 16 text
windows per image, spreads them across very long posts, and reports coverage.
An uncovered part of a post may mention the image. Long results without a
separate cutoff, or with incomplete coverage, remain provisional. A high
similarity in one window does not prove the whole post matches the image.
