# Layer 1 data repair and evaluation

This workflow changes the binary AI-involvement detector. H means verified human writing (label `0`); L1–L4 each mean some AI involvement (label `1`). The binary detector cannot identify *which* L category a post belongs to. The `variant` field is retained so that recall for each category can be reported separately.

## 1. Freeze and inspect the current artifacts

Work in a clean branch and record the commit ID. From `backend/`, run:

```bash
python scripts/audit_layer1_data.py > layer1_before.txt
```

The audit prints SHA256 values, record counts by source/label/variant, length buckets, and IDs/groups crossing existing splits. Existing split files predate `source_group_id`, so missing-group counts are expected *before* regeneration. Inspect the raw RedNote field names and confirm `desc` holds human post bodies while `note_content` holds generated post bodies. Confirm the AnxForever label mapping against the dataset's documentation; do not infer the meaning of `0` and `1` from counts. CHASM `ad_label` is an advertising label, not proof of human authorship.

## 2. Prepare verified H/L groups

Supply a UTF-8 JSONL file outside the repository (for example `data/private/reviewed_hl.jsonl`). Each line is one verified post. Each original post and all derivatives share one stable `source_group_id`:

```json
{"id":"case-001-H","source_group_id":"case-001","variant":"H","label":0,"human_subtype":"organic","text":"完整的真人帖子标题和正文"}
{"id":"case-001-L1","source_group_id":"case-001","variant":"L1","label":1,"text":"同一原帖衍生的完整文本"}
```

Provide L2, L3, and L4 for that group as well. A verified human post can also enter as a standalone H group when no derivatives exist. The importer rejects partially supplied derivative sets, duplicate variants, missing identifiers, and mismatched labels. An H post can use `human_subtype: "organic"` or `"ad"` after manual review. Add varied, full-length human recommendation posts and verified human-written ads, including commercial language likely to trigger false alarms. Do not relabel CHASM ads as human simply because they are ads. Check provenance, consent, and whether AI was used in drafting before giving an H label.

Reserve a **new, untouched** collection of complete H/L groups in a separate JSONL file such as `data/private/final_hl.jsonl`. Do not pass that file to preparation, splitting, calibration, or threshold selection. The already inspected 46 H posts are useful diagnostic material but cannot serve as a fresh final test after they influenced changes.

## 3. Rebuild on the branch

Activate the backend environment and install `requirements.txt` if necessary. `prepare_datasets.py` downloads AnxForever and CHASM data, so it needs access to Hugging Face. From `backend/`:

```bash
python scripts/prepare_datasets.py --reviewed-data data/private/reviewed_hl.jsonl
python scripts/create_splits.py
python scripts/audit_layer1_data.py > layer1_after.txt
python train_detector.py
python scripts/calibrate_detector.py
python scripts/select_layer1_threshold.py --max-human-fpr 0.05
python scripts/evaluate_locked_test.py
```

`create_splits.py` uses four splits: fit (50% of groups), dev (15%), calibration (15%), and locked test (20%). It joins rows with the same `source_group_id`, exact text, or very high text similarity before assigning groups. The original H and its four derivatives therefore remain together. Review `layer1_after.txt`: every split row needs a group ID, no group or ID may cross splits, and human text-length distributions should reflect full bodies. Review the fit/dev/calibration/test counts for each variant and source; small subgroups give uncertain rates. The split proportions apply to groups, not necessarily to rows or every variant. Stop if any important subgroup is absent from the locked test.

The base TF-IDF/logistic regression model is fit on `fit`. The sigmoid calibrator is fit on `calibration`. The threshold selector uses `dev` to maximize AI recall while keeping measured human false-positive rate at or below the chosen limit. The `0.05` limit is an example policy target, not a promised real-world rate. Do not change the model or threshold in response to the locked-test or final-test results; if you do, collect a new untouched final test. The API reads the threshold from the saved calibrated model bundle after restart.

The retraining writes `models/authorship_provenance.json` and embeds fit, dev, calibration, and base-model hashes in the calibrated bundle. The old model predates this metadata, so its exact training files cannot be established solely from the model artifact. Run the audit again after training and keep its output with the results.

## 4. Evaluate and report

### Compare Chinese pretrained candidates before fine-tuning

First compare the current detector with `yuchuantian/AIGC_detector_zhv3` and `yuchuantian/AIGC_detector_zhv3short` on the same *diagnostic* H/L JSONL file. From `backend/`:

```bash
python scripts/benchmark_layer1_candidates.py --input data/private/diagnostic_hl.jsonl --output data/private/benchmark_before.json
```

This downloads two Chinese BERT classification weights (about 409 MB each) on first use. Each is loaded sequentially. The public author demo maps class 0 to human and class 1 to AI and truncates to 512 tokens; this script uses that mapping, records token truncation, and reports H false positives and each L group's detection rate. The scores from different models are not calibrated to each other. A fixed 0.5 comparison is a first screen, not a final operating-point decision. Run `--models zhv3` to load one model if resources are limited. The older locked test can be used for a read-only diagnostic comparison, but once it guides model choice it is no longer an untouched final test.

If a candidate performs better on verified human posts, fine-tune its *existing checkpoint* using only the corrected `fit.jsonl`, select settings with `dev.jsonl`, calibrate with `calibration.jsonl`, then evaluate on an untouched test. Keep its tokenizer with that checkpoint and repeat the 512-token truncation audit. The official repository supplies a `train.py`, but its published training commands demonstrate older English experiments; do not assume those commands exactly reproduce Chinese V3. Keep the current baseline available until the candidate is validated on the full domain test.

From `backend/`, run one candidate at a time. Keep model outputs in `../model_runs/` (ignored by Git) rather than overwriting `models/authorship_calibrated.joblib`:

```bash
python scripts/fine_tune_layer1.py --base zhv3 --output-dir ../model_runs/zhv3-run1 --epochs 2 --batch-size 4 --learning-rate 2e-5
python scripts/calibrate_layer1_candidate.py --model-dir ../model_runs/zhv3-run1 --max-human-fpr 0.05
python scripts/evaluate_layer1_candidate.py --model-dir ../model_runs/zhv3-run1
```

For the short model, use `--base zhv3short --output-dir ../model_runs/zhv3short-run1` and replace the model-dir in subsequent commands. Training saves the best epoch based on development-set AI recall subject to a human false-positive limit. It uses balanced class weights and truncates at 512 tokens; the selected epoch and hashes are written to `training_metadata.json`. Calibration trains a separate sigmoid on the calibration split and selects the decision threshold on dev. Calibration and evaluation reject changed splits and source-group overlap. The final command above uses the internal locked test; use `--test-file data/private/final_hl.jsonl` for a *new* fully independent set after model selection. Never train or tune on either test result. If memory is tight on a MacBook Air, start with `--batch-size 2`; CPU fallback may be slow. Confirm PyTorch reports MPS availability if you expect GPU training.

### Final untouched test

```bash
python scripts/evaluate_locked_test.py --test-file data/private/final_hl.jsonl
```

The external evaluation checks that no ID or `source_group_id` appears in any of the four internal splits and prints SHA256 hashes for the model and test file. Also check near duplicates and shared source posts manually across external data. Report the *count and denominator* for H false positives (`FP / all H`) and for each of L1–L4 detection (`TP / all posts in that category`), plus human organic/advertising breakdown and text lengths. A single overall accuracy obscures false alarms. Keep a record of the dataset provenance, label review, Git commit, file hashes, threshold, and Python/scikit-learn versions.

Only after reviewing the metrics, restart the backend, exercise a few known human and AI examples through the API, and review the generated-file diff before committing. Do not present the displayed calibrated score as a literal percentage of AI-written characters.
