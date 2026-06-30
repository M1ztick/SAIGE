# SAIGE — Project Structure

## Directory Layout

```
SAIGE/
├── worker/                          # Cloudflare Worker — edge inference + experience logging
│   ├── worker.ts                    # Main API handler (3 endpoints)
│   ├── harm_detection.ts            # 4-dimension harm scoring engine
│   ├── buddhist_principles.ts       # 5-principle Buddhist alignment scorer
│   ├── wrangler.toml                # Cloudflare deployment config (D1 binding)
│   ├── package.json                 # Worker build config (wrangler dev/deploy)
│   └── init-db.sh                   # Database initialization script
│
├── local-trainer/                   # Training pipeline — data conversion + fine-tuning
│   ├── annotations/                 # Human-annotated DPO preference pairs
│   │   ├── saige-coverage-schema.json    # Annotation schema
│   │   └── saige-rs-001..008.json        # 8 annotated Right Speech scenarios
│   ├── generate_dpo_pairs.py        # Builds DPO pairs from annotations + Claude API
│   ├── scorers.py                   # Shared scoring utilities for DPO pipeline
│   ├── dpo_pairs.jsonl              # Generated DPO training pairs (JSONL)
│   ├── dpo_pairs.json               # Same pairs in JSON array format
│   ├── dpo_pairs_pretty.json        # Pretty-printed version
│   ├── .dpo_cache.json              # Cache to avoid re-generating pairs
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Training pipeline documentation
│
├── sql/                             # Database seed data
│   ├── seed_scenarios.sql           # Training scenarios (difficulty 1–4)
│   ├── seed_conversational_calibration.sql  # 23 calibration scenarios
│   └── seed-simple.sql              # Simplified seed for quick setup
│
├── Documents/                       # Design and milestone documentation
│   ├── README.md                    # Architecture philosophy and system design
│   ├── SETUP_LOG.md                 # Execution log and current project status
│   ├── BUDDHIST-INTEGRATION.md      # Buddhist scoring system details
│   ├── RL-TO-SFT-PIPELINE.md        # v1 pipeline design (historical)
│   ├── FIXING-SPAZZY-TINYLLAMA.md   # Calibration training strategy
│   ├── README_dataset.md            # Training dataset card
│   └── qwen3-vl:2b-instruct-buddhist-training.txt  # Qwen3 training notes
│
├── Buddhist Reference Archive/      # Legacy reference implementations
│   ├── evaluate_buddhist_ethics.py  # Ethics evaluation suite
│   ├── convert_rl_to_sft.py         # v1 converter (superseded)
│   └── buddhist_worker_lora.js      # LoRA deployment reference for Cloudflare
│
├── data/                            # Supplementary datasets
│   └── iran_war_profiteering_dataset.json  # Domain-specific test dataset
│
├── schema.sql                       # SQLite database schema (4 tables)
├── saige.db                         # Local SQLite database
├── init_db.py                       # Local database initializer
├── pull_remote_data.py              # Syncs experiences from deployed worker
├── load-calibration-scenarios.sh    # Loads calibration scenarios into DB
├── test_request.json                # Sample API request for testing
├── test-buddhist-integration.ts     # Integration test for Buddhist scoring
└── README.md                        # Project root documentation
```

---

## Core Components and Relationships

### 1. Worker (Cloudflare Edge)
The worker is the live inference and data collection layer.

- `worker.ts` — 3 REST endpoints:
  - `GET /api/get-scenario` — returns random scenario from D1, filtered by difficulty
  - `POST /api/simulate-outcome` — scores an AI response, stores experience
  - `GET /api/stats` — returns 24-hour training statistics
- `harm_detection.ts` — imported by worker.ts; scores responses across 4 harm dimensions
- `buddhist_principles.ts` — imported by harm_detection.ts (or worker.ts); scores 5 Buddhist principles

### 2. Database (SQLite / Cloudflare D1)
Four tables defined in `schema.sql`:

| Table | Purpose |
|-------|---------|
| `scenarios` | Training prompts with difficulty, harm type, person_state, critical_info |
| `experiences` | Logged AI responses with harm scores, Buddhist scores, learned lessons |
| `causal_patterns` | Abstracted patterns the AI has identified |
| `model_versions` | Checkpoint tracking with performance metrics |

JSON columns: `person_state`, `facts`, `critical_info`, `harm_breakdown`, `buddhist_scores`, `example_ids`

### 3. DPO Annotation Pipeline (local-trainer)

```
annotations/saige-rs-*.json  →  generate_dpo_pairs.py  →  dpo_pairs.jsonl
```

- Each annotation file contains a prompt, chosen response, rejected response, and per-dimension scores
- `generate_dpo_pairs.py` uses the Anthropic Claude API to evaluate and generate response pairs
- `scorers.py` provides shared scoring logic used by the generator
- Output format: JSONL with fields — prompt, chosen, rejected, record_id, path_factor, canonical_id, pair_type, score metadata

### 4. Training Pipeline (superseded scripts referenced in README)
The README references `saige_to_sft_v2.py` and `train_local.py`, which were the v2 pipeline. The current active pipeline has shifted to DPO-based training. The `local-trainer/` directory now contains the DPO tooling.

---

## Architectural Patterns

### Edge-First Data Collection
All live inference and experience logging happens on Cloudflare Workers + D1. Local scripts pull data down via `pull_remote_data.py` for offline training.

### Separation of Scoring Concerns
- Harm detection and Buddhist principle scoring are separate TypeScript modules in the worker
- Python-side scoring in `scorers.py` mirrors the worker-side logic for offline evaluation

### JSON-in-SQL Pattern
Complex structured fields (person_state, buddhist_scores, harm_breakdown) are stored as JSON strings in SQLite TEXT columns and parsed at application boundaries with `safeJsonParse()`.

### Annotation-Driven DPO
Rather than purely programmatic preference generation, SAIGE uses human-authored annotation files (per-response dimensional scores + flaws) as the ground truth source for DPO pairs. These are keyed to canonical Buddhist texts as philosophical anchors.

### Caching for Expensive Operations
`generate_dpo_pairs.py` maintains `.dpo_cache.json` to avoid re-calling the Claude API for already-processed annotations.
