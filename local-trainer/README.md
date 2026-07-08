# SAIGE Local Trainer — DPO Annotation Pipeline

This directory contains the annotation system and DPO fine-tuning pipeline for SAIGE. The goal is to train a model whose preferred behavior — honest, compassionate, non-divisive, well-calibrated speech — is unconditional: present regardless of what system prompt is active or absent.

The pipeline has three stages:

```
annotations/saige-rs-*.json
  → generate_dpo_pairs.py       Claude API: expand prompts → generate candidates → judge
  → dpo_pairs.jsonl
  → diversify_prompts.py        stratify by system prompt condition: rs / generic / none
  → dpo_pairs_diversified.jsonl
  → SAIGE_DPO_Training.ipynb    Qwen2.5-3B + QLoRA on Colab T4
```

---

## Annotation Records

Each annotation file in `annotations/` maps a canonical Buddhist text to a set of concrete AI behavior targets. The annotation is the unit of work — it defines what the model should learn, what failure modes to contrast against, and what questions the LLM judge uses to score responses.

### Schema

The fields that drive the pipeline:

| Field | Purpose |
|-------|---------|
| `canonical_id` | Buddhist text reference (e.g. `SN 45.8`, `MN 58`) |
| `core_principle` | One-sentence statement of what the record teaches |
| `behavior_targets` | Specific behaviors the chosen response should exhibit |
| `failure_modes` | Specific behaviors the rejected response should exhibit |
| `unsafe_misreadings` | Named misreadings used to generate misreading pairs (see below) |
| `example_prompt_types` | Descriptions expanded into concrete user messages by Claude |
| `evaluation_questions` | Per-dimension scoring questions sent to the LLM judge |
| `annotation_status` | Lifecycle stage — controls which records are processed |

### Status Lifecycle

```
pending → draft → committed → reviewed
```

`generate_dpo_pairs.py` processes `draft` and `committed` records by default. `pending` records exist in the coverage schema but have not been written yet. Pass `--status committed` to restrict to fully reviewed records only.

### Current Records

All 12 active records cover Right Speech (`rs`). The coverage schema (`saige-coverage-schema.json`) maps out the full Noble Eightfold Path — Right Intention, Right View, Right Action, and others — as future annotation targets.

| ID | Canonical | Title | Status |
|----|-----------|-------|--------|
| saige-rs-001 | SN 45.8 | Definition of Right Speech | draft |
| saige-rs-002 | MN 58 | Speech Should Be True, Beneficial, and Timely | draft |
| saige-rs-003 | MN 61 | Reflect Before, During, and After Speech | draft |
| saige-rs-004 | AN 5.198 | Five Factors of Well-Spoken Speech | draft |
| saige-rs-005 | MN 21 | The Saw Simile — Speech Under Provocation | draft |
| saige-rs-006 | AN 4.183 | Four Types of Persons in Terms of Speech | draft |
| saige-rs-007 | Dhp 1-2 | Mind as Forerunner of Speech and Action | draft |
| saige-rs-008 | Sn 3.3 | The Sword Simile — Harsh Speech Cuts Both Ways | draft |
| saige-rs-009 | AN 3.68 | Three Kinds of People in Their Use of Speech | draft |
| saige-rs-010 | AN 10.69 | Ten Unwholesome Courses of Action — Speech Portion | draft |
| saige-rs-011 | MN 139 | Non-Quarrelsome Exposition — Speaking Without Dispute | draft |
| saige-rs-012 | AN 5.157 | Idle Chatter as a Distinct Failure Mode | draft |

---

## Pair Types

The pipeline produces two kinds of DPO pairs per record.

### Ranked pairs

For each `example_prompt_type`, Claude expands the description into concrete user messages, then generates multiple candidate responses — some under the SAIGE system prompt, some under a generic baseline. All candidates are scored by an LLM judge using the record's `evaluation_questions`. The highest and lowest scoring candidates become the `chosen` and `rejected` sides of the pair.

A pair is only written if `score_delta >= 1`. Most pairs sit at delta 1–3; delta ≥ 4 is strong signal.

### Misreading pairs

Each `unsafe_misreading` in the annotation names a specific way the principle can be misapplied — for example, `"mistaking politeness for truthfulness"` or `"replacing substance with defensive hedging"`. For each misreading, a persona-injected system prompt causes Claude to generate a response that enacts that failure. The best ranked candidate from the same record becomes the `chosen` side; the misreading response becomes `rejected`.

Misreading pairs are the higher-signal half of the dataset. They contrast a genuinely good response against a response that fails in a specific, named way — not just a weaker version of the same thing.

---

## Step 1 — Generate DPO Pairs

Requires `ANTHROPIC_API_KEY` in the environment.

```bash
cd local-trainer

# Default run — processes draft + committed records
python generate_dpo_pairs.py

# Preview without API calls
python generate_dpo_pairs.py --dry-run

# More pairs per record
python generate_dpo_pairs.py --prompts-per-type 4 --candidates-per-prompt 4

# Committed records only
python generate_dpo_pairs.py --status committed

# Regenerate everything, ignoring cache
python generate_dpo_pairs.py --no-cache
```

The generator caches all API calls to `.dpo_cache.json`. Re-runs are cheap — only new or changed records trigger API calls. Cache keys incorporate a hash of the judge prompt and scoring rubric, so editing either self-invalidates stale scores.

Output: `dpo_pairs.jsonl` — one JSON object per line, appended on each run.

### Scoring and anti-inflation

The LLM judge uses an anchored rubric with explicit calibration rules to resist score inflation. After scoring, the pipeline deterministically reconciles the judge's overall score against its per-dimension scores:

```
reconciled = min(judge_overall, round(0.6 × worst_subscore + 0.4 × mean_subscore))
```

The worst dimension is weighted 60%. A single major failure pulls the overall down regardless of how strong the other dimensions are.

---

## Step 2 — Diversify System Prompt Conditions

```bash
python diversify_prompts.py
```

This takes `dpo_pairs.jsonl` and produces `dpo_pairs_diversified.jsonl` with each pair assigned one of three system prompt conditions, cycling round-robin within each record:

| Condition | System message |
|-----------|---------------|
| `rs` | Full SAIGE Right Speech prompt |
| `generic` | `"You are a helpful AI assistant."` |
| `none` | No system message |

The `prompt_condition` field is added to each pair for inspection. The training goal is that the model learns preferred behavior is unconditional — not a response to the SAIGE prompt being present. If `adapter + generic prompt` looks close to `adapter + RS prompt` in the inference ablation, the training worked.

Note on `--min-delta`: at the current dataset size (~85 pairs), most pairs are at delta=1. Filtering to `--min-delta 2` drops to ~21 pairs — too few. Apply this filter only when the dataset exceeds ~300 pairs.

---

## Step 3 — Fine-Tune (Colab)

Open `SAIGE_DPO_Training.ipynb` in Google Colab with a T4 GPU.

The notebook:
1. Loads `M1ztyk/SAIGE-right-speech-dpo` from HuggingFace (the diversified pairs)
2. Loads `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 (fits T4 with ~4GB headroom)
3. Trains with DPO + QLoRA for 3 epochs
4. Pushes the adapter to `M1ztyk/SAIGE-dpo-v2`

Key hyperparameters:
- `beta=0.1` — how far the policy can drift from the reference model
- `lr=5e-7` — DPO is sensitive to LR; lower than SFT
- `lora_rank=16`, `lora_alpha=32`, all projection layers targeted

After training, run the 2×2 ablation in `SAIGE_DPO_Inference.ipynb`:

```
adapter + RS prompt      adapter + generic prompt
base    + RS prompt      base    + generic prompt
```

The goal: `adapter + generic` should look close to `adapter + RS`.

---

## Uploading a New Dataset Version

```bash
python upload_dataset.py
```

Uploads `dpo_pairs_diversified.jsonl` to `M1ztyk/SAIGE-right-speech-dpo` on HuggingFace. Requires `HF_TOKEN` in the environment with write scope.

---

## Files

| File | Purpose |
|------|---------|
| `annotations/saige-rs-*.json` | Annotation records (12 Right Speech) |
| `annotations/saige-coverage-schema.json` | Full coverage plan across the Noble Eightfold Path |
| `generate_dpo_pairs.py` | Annotation → DPO pairs via Claude API |
| `scorers.py` | Shared scoring utilities (calibration, coherence, Buddhist weighting) |
| `diversify_prompts.py` | Stratify pairs by system prompt condition |
| `dpo_pairs.jsonl` | Raw generated pairs |
| `dpo_pairs_diversified.jsonl` | Pairs with `prompt_condition` assigned — this is the training input |
| `.dpo_cache.json` | API call cache (do not commit) |
| `SAIGE_DPO_Training.ipynb` | Colab fine-tuning notebook |
| `SAIGE_DPO_Inference.ipynb` | Colab inference and ablation notebook |
| `upload_dataset.py` | Push diversified pairs to HuggingFace |
| `requirements.txt` | Python dependencies |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, an `ANTHROPIC_API_KEY` for pair generation, and a HuggingFace token with write scope for dataset upload and model push.
