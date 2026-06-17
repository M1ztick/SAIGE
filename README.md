# SAIGE — Systems-Aware Independently-Governing Ethics

An AI alignment framework that teaches ethical reasoning through **experiential learning** and **Buddhist wisdom principles**, rather than rule-based programming.

> Live worker: `https://buddhist-ai-worker.mistykmedia.workers.dev`

---

## What SAIGE Does

SAIGE trains language models to reason ethically by giving them experience — thousands of curated conversational scenarios scored across two dimensions:

- **Harm avoidance** (deception, harshness, omission, manipulation)
- **Positive principle embodiment** (ahimsa, sacca, karuna, panna, upekkha)

The model improves in a continuous loop: deploy → collect experiences → score and filter → fine-tune → redeploy.

---

## Project Structure

```
S-A-I-G-E/
├── worker/                  # Cloudflare Worker — live inference + experience logging
│   ├── worker.ts            # Main API handler (get-scenario, simulate-outcome)
│   ├── harm_detection.ts    # 4-dimension harm scoring
│   ├── buddhist_principles.ts  # 5-principle Buddhist alignment scoring
│   └── wrangler.toml        # Cloudflare deployment config (D1 binding)
│
├── local-trainer/           # Training pipeline — data conversion + fine-tuning
│   ├── saige_to_sft_v2.py   # Experience → SFT training data converter
│   ├── train_local.py       # LoRA/QLoRA fine-tuning (TinyLlama, Mistral, etc.)
│   ├── trainer.js           # Experience collection script (calls worker API)
│   ├── train_pipeline.sh    # End-to-end pipeline orchestrator
│   └── README.md            # Full training pipeline documentation
│
├── sql/                     # Database seed data
│   ├── seed_scenarios.sql               # Training scenarios (difficulty 1–4)
│   └── seed_conversational_calibration.sql  # 23 calibration scenarios
│
├── Documents/               # In-depth design and milestone documentation
│   ├── README.md            # Architectural philosophy and system design
│   ├── SETUP_LOG.md         # Execution log and current project status
│   ├── BUDDHIST-INTEGRATION.md      # Buddhist scoring system (Priority 1)
│   ├── RL-TO-SFT-PIPELINE.md        # Training pipeline design (Priority 2)
│   ├── FIXING-SPAZZY-TINYLLAMA.md   # Calibration training strategy
│   └── README_dataset.md            # Training dataset card
│
├── Buddhist Reference Archive/  # Legacy reference implementations
│   ├── evaluate_buddhist_ethics.py  # Ethics evaluation suite
│   └── convert_rl_to_sft.py         # v1 converter (superseded by v2)
│
├── schema.sql               # SQLite database schema
├── saige.db                 # Local database (scenarios + experiences)
├── init_db.py               # Local database initializer
└── pull_remote_data.py      # Sync experiences from deployed worker
```

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                  Continuous Improvement Loop                 │
│                                                              │
│   Cloudflare Worker                                          │
│   ├─ Receives scenario from D1 database                     │
│   ├─ Generates AI response (TinyLlama via Ollama)           │
│   ├─ Scores: harm (4 dims) + Buddhist alignment (5 dims)    │
│   └─ Logs experience to database                            │
│          ↓                                                   │
│   saige_to_sft_v2.py                                        │
│   ├─ Cleans: typos, placeholders, AI-prefix artifacts       │
│   ├─ Scores: calibration, coherence, composite              │
│   ├─ Filters: harm < 0.25, composite ≥ 5.0                  │
│   └─ Outputs: CSV in Mistral / ChatML / Llama3 format       │
│          ↓                                                   │
│   train_local.py                                             │
│   ├─ LoRA/QLoRA fine-tuning on filtered examples            │
│   └─ Saves adapter weights                                  │
│          ↓                                                   │
│   Deploy improved model → collect better experiences →  ↺   │
└──────────────────────────────────────────────────────────────┘
```

---

## Buddhist Ethics Scoring

Five principles form the positive ethics backbone, scored 0–10 per response:

| Principle | Weight | Meaning |
|-----------|--------|---------|
| Ahimsa (non-harm) | 25% | Avoid causing suffering |
| Sacca (truthfulness) | 20% | Honest, accurate communication |
| Karuna (compassion) | 25% | Genuine care for the other person |
| Panna (wisdom) | 20% | Contextual understanding, root-cause thinking |
| Upekkha (equanimity) | 10% | Calm, tone-matched, non-reactive |

Responses are filtered by a **composite score** blending all quality dimensions:

```
composite = Buddhist×0.35 + Calibration×0.30 + Coherence×0.20 + (1−Harm)×10×0.15
```

---

## Current Dataset

| Metric | Value |
|--------|-------|
| Total experiences collected | 150 |
| High-quality examples (passed filter) | 91 |
| Average harm score | 0.055 |
| Average Buddhist score | 7.24 |
| Alignment distribution | 5.5% excellent / 94.5% good |
| Difficulty spread | Levels 1–4 |

Training data: [`local-trainer/saige_training_data_v2.csv`](local-trainer/saige_training_data_v2.csv)

---

## Quick Start

### Generate Training Data

```bash
cd local-trainer

# Convert experiences from the database to SFT format
python saige_to_sft_v2.py \
    --db ../saige.db \
    --output saige_training_data_v2.csv \
    --include-gold

# Or run the full pipeline (collection → conversion → recommendations)
./train_pipeline.sh
```

### Fine-Tune Locally

```bash
# TinyLlama 1.1B — ~4GB VRAM with 4-bit
python train_local.py \
    --data saige_training_data_v2.csv \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --use-4bit

# Mistral 7B — ~8GB VRAM with 4-bit
python train_local.py \
    --data saige_training_data_v2.csv \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --use-4bit --lora-rank 32
```

### Collect New Experiences

```bash
cd local-trainer
node trainer.js 100   # Collect 100 training episodes from the live worker
```

### Deploy Worker

```bash
cd worker
wrangler deploy
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Documents/README.md](Documents/README.md) | Full architectural design, wisdom principles, system philosophy |
| [Documents/SETUP_LOG.md](Documents/SETUP_LOG.md) | Execution log, current project status, deployment notes |
| [Documents/BUDDHIST-INTEGRATION.md](Documents/BUDDHIST-INTEGRATION.md) | Buddhist scoring system implementation details |
| [Documents/RL-TO-SFT-PIPELINE.md](Documents/RL-TO-SFT-PIPELINE.md) | Original v1 pipeline design (historical) |
| [Documents/FIXING-SPAZZY-TINYLLAMA.md](Documents/FIXING-SPAZZY-TINYLLAMA.md) | Calibration training strategy for verbosity control |
| [Documents/README_dataset.md](Documents/README_dataset.md) | Training dataset card and quality metrics |
| [local-trainer/README.md](local-trainer/README.md) | Training pipeline usage guide (v2, current) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Edge inference | Cloudflare Workers (TypeScript) |
| Database | Cloudflare D1 / SQLite |
| Local LLM | Ollama + TinyLlama 1.1B |
| Fine-tuning | PyTorch, HuggingFace Transformers, TRL, PEFT |
| Quantization | bitsandbytes (4-bit / 8-bit QLoRA) |
| Training formats | Mistral, ChatML, Llama3, Alpaca |

---

## Roadmap

- [x] Buddhist principle scoring (Priority 1)
- [x] RL-to-SFT training pipeline (Priority 2)
- [x] Conversational calibration training
- [x] v2 converter with composite scoring
- [ ] Buddhist ethics evaluation suite (Priority 3)
- [ ] First fine-tuned model checkpoint
- [ ] Continuous collection → retrain loop
