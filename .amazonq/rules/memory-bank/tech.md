# SAIGE — Technology Stack

## Languages

| Layer | Language | Version |
|-------|----------|---------|
| Edge Worker | TypeScript | Compatible with Cloudflare Workers runtime |
| Training Pipeline | Python | 3.8+ |
| Experience Collector | JavaScript (Node.js) | Node 18+ (assumed for Cloudflare ecosystem) |
| Database | SQL (SQLite / D1) | SQLite-compatible |

---

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Edge runtime | Cloudflare Workers |
| Edge database | Cloudflare D1 (SQLite-compatible) |
| Local database | SQLite (`saige.db`) |
| Deployment tool | Wrangler v3+ |
| Local LLM inference | Ollama + TinyLlama 1.1B |
| AI response generation (DPO) | Anthropic Claude API (`anthropic>=0.40.0`) |

---

## Python Dependencies (`local-trainer/requirements.txt`)

### Core ML
- `torch>=2.0.0` — PyTorch base
- `transformers>=4.35.0` — HuggingFace model loading and tokenization
- `datasets>=2.14.0` — Dataset handling

### Training
- `trl>=0.7.0` — Transformer Reinforcement Learning (SFT, DPO trainers)
- `peft>=0.6.0` — Parameter-Efficient Fine-Tuning (LoRA/QLoRA adapters)
- `accelerate>=0.24.0` — Multi-GPU / mixed-precision training
- `bitsandbytes>=0.41.0` — 4-bit and 8-bit quantization (QLoRA)

### Data & Utilities
- `pandas>=2.0.0` — CSV/dataframe operations
- `numpy>=1.24.0` — Numerical operations
- `tensorboard>=2.14.0` — Training visualization
- `anthropic>=0.40.0` — Claude API client (DPO pair generation)

### Optional
- `scipy`, `scikit-learn` — Advanced analysis
- `wandb` — Weights & Biases experiment tracking

---

## Node/Worker Dependencies (`worker/package.json`)

- `wrangler ^3.0.0` (devDependency) — Cloudflare Worker build and deployment CLI

Worker name: `buddhist-ai-worker`
Compatibility date: `2024-01-18`
D1 binding: `DB` → database `buddhist-ai-training` (ID: `33dac81c-4241-4f0a-a2b7-03ea13af5370`)

---

## Supported Model Targets

| Model | Size | VRAM (full) | VRAM (4-bit) | Notes |
|-------|------|-------------|--------------|-------|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | ~8GB | ~4GB | Default, fastest |
| Mistral-7B-Instruct-v0.2 | 7B | ~28GB | ~8GB | Better quality |
| Qwen3-VL-2B-Instruct | 2B | — | — | Referenced in Documents/ |

---

## Training Formats

| Format | Template |
|--------|----------|
| ChatML / TinyLlama | `<\|system\|>...<\/s><\|user\|>...<\/s><\|assistant\|>...<\/s>` |
| Mistral | `<s>[INST] ... [/INST] ...</s>` |
| Llama 3 | `<\|begin_of_text\|><\|start_header_id\|>...<\|eot_id\|>` |
| Alpaca | `### Instruction:\n...\n### Response:\n...` |

---

## Development Commands

### Worker
```bash
cd worker

# Local development
npm run dev           # wrangler dev --local

# Deploy to Cloudflare
npm run deploy        # wrangler deploy

# Initialize D1 database
bash init-db.sh
```

### Training Pipeline
```bash
cd local-trainer

# Install Python deps
pip install -r requirements.txt

# Generate DPO pairs from annotations
python generate_dpo_pairs.py

# (Legacy) Convert DB experiences to SFT CSV
python saige_to_sft_v2.py --db ../saige.db --output training_data.csv --include-gold

# (Legacy) Fine-tune with LoRA
python train_local.py --data training_data.csv --use-4bit

# Collect new experiences from live worker
node trainer.js 100
```

### Database
```bash
# Initialize local DB
python init_db.py

# Load calibration scenarios
bash load-calibration-scenarios.sh

# Sync from remote worker
python pull_remote_data.py

# Inspect
sqlite3 saige.db "SELECT COUNT(*) FROM experiences"
sqlite3 saige.db "SELECT COUNT(*) FROM scenarios"
```

### Full Pipeline (Legacy)
```bash
cd local-trainer
./train_pipeline.sh

# With custom env vars:
TRAINING_EPISODES=100 MAX_HARM=0.2 MIN_COMPOSITE=6.0 ./train_pipeline.sh
```

---

## Environment Variables (train_pipeline.sh)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `../saige.db` | Local database path |
| `WORKER_URL` | `http://localhost:8787` | SAIGE worker URL |
| `TRAINING_EPISODES` | `50` | Episodes to collect |
| `MAX_HARM` | `0.25` | Maximum harm threshold |
| `MIN_COMPOSITE` | `5.0` | Minimum composite quality score |
| `MIN_CALIBRATION` | `2.0` | Minimum calibration score |
| `FORMAT` | `mistral` | Output training format |

---

## LoRA Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--lora-rank` | 16 | 32 recommended for Mistral |
| `--lora-alpha` | 32 | — |
| `--learning-rate` | 2e-4 | Increase to 5e-4 if loss stalls |
| `--epochs` | 3 | Increase to 5 for small datasets |
| `--batch-size` | 4 | Reduce to 1–2 if OOM |
