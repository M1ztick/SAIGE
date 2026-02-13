# SAIGE Local Trainer - RL-to-SFT Training Pipeline

Complete training pipeline for converting SAIGE's experiential learning data into fine-tuned language models.

## Overview

This pipeline automates the process of:
1. **Collecting experiences** via interaction with SAIGE worker
2. **Cleaning, scoring, and filtering responses** using multi-dimensional quality metrics
3. **Converting to SFT format** (Mistral, ChatML/TinyLlama, Llama3, or Alpaca)
4. **Fine-tuning models** locally or in the cloud

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Install dependencies
pip install -r requirements.txt

# For local fine-tuning, you'll need:
# - CUDA-capable GPU (8GB+ VRAM recommended)
# - Or use Google Colab for cloud training
```

### Basic Usage

```bash
# Full pipeline: collect → convert → prepare
./train_pipeline.sh

# Or run steps individually:

# Step 1: Collect experiences (optional if you already have data)
node trainer.js 100

# Step 2: Convert to SFT format with cleaning and scoring
python saige_to_sft_v2.py --db ../saige.db --output training_data_v2.csv --include-gold

# Step 3: Fine-tune (if you have GPU)
python train_local.py --data training_data_v2.csv
```

---

## Scripts

### 1. `saige_to_sft_v2.py` - Experience → Training Data Converter

Queries the SAIGE database, cleans generation artifacts, scores quality across multiple dimensions, and converts high-quality examples to SFT format.

**Usage:**
```bash
python saige_to_sft_v2.py \
    --db ../saige.db \
    --output saige_training_data_v2.csv \
    --format mistral \
    --max-harm 0.25 \
    --min-composite 5.0 \
    --min-calibration 2.0 \
    --include-gold
```

**Parameters:**
- `--db`: Path to SAIGE database (default: `saige.db`)
- `--output`: Output CSV file (default: `saige_training_data_v2.csv`)
- `--format`: Training format — `mistral`, `chatml`, `llama3`, or `alpaca` (default: `mistral`)
- `--max-harm`: Maximum harm score to include, 0-1 (default: `0.25`)
- `--min-composite`: Minimum composite quality score, 0-10 (default: `5.0`)
- `--min-calibration`: Minimum calibration score, 0-10 (default: `2.0`)
- `--best-per-scenario`: Keep only the single best example per scenario
- `--include-gold`: Add hand-crafted gold-standard expected responses
- `--include-negatives`: Add harmful/miscalibrated examples (for DPO/contrastive training)
- `--diagnostics-only`: Print quality diagnostics without writing output

**Composite Scoring:**

Each example is scored across four dimensions, then ranked by a weighted composite:

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| Buddhist | 35% | Ahimsa, Sacca, Karuna, Panna, Upekkha alignment |
| Calibration | 30% | Response length appropriate to prompt complexity |
| Coherence | 20% | Structural quality, absence of repetition/meta-commentary |
| Harm | 15% | Inverse of actual harm score |

**Output:**
CSV file with columns:
- `text`: Formatted training example (in the requested format)
- `harm_score`: Harm score (0-1, lower is better)
- `buddhist_alignment`: Alignment level (low/moderate/good/excellent)
- `weighted_score`: Adjusted Buddhist weighted score (0-10)
- `calibration_score`: Response-length appropriateness (0-10)
- `coherence_score`: Structural quality (0-10)
- `composite_score`: Blended ranking score (0-10)
- `difficulty`: Scenario difficulty level (1-5)
- `scenario_id`: Original scenario ID
- `experience_id`: Experience record ID (prefix `gold_` or `neg_` for injected examples)
- `is_gold`: True for hand-crafted ideal responses
- `is_negative`: True for DPO contrastive examples

**Companion output:** `*_diagnostics.csv` — per-example cleaning audit (typos fixed, placeholders removed, calibration/coherence scores, rejection reasons).

---

### 2. `train_local.py` - Local Fine-Tuning with LoRA/QLoRA

Fine-tunes TinyLlama, Mistral, or other models using the training data.

**Usage:**
```bash
# TinyLlama (1.1B) - good for testing, runs on smaller GPUs
python train_local.py \
    --data saige_training_data_v2.csv \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16

# Mistral (7B) with 4-bit quantization - better quality, needs 16GB+ VRAM
python train_local.py \
    --data saige_training_data_v2.csv \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --use-4bit \
    --epochs 3 \
    --batch-size 2 \
    --lora-rank 32
```

**Parameters:**
- `--data`: Training data CSV file (required)
- `--model`: Model name or path (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `--use-4bit`: Use 4-bit quantization (QLoRA) for memory efficiency
- `--use-8bit`: Use 8-bit quantization
- `--lora-rank`: LoRA rank (default: 16, higher = more capacity but slower)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size per device (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--output-dir`: Output directory (default: `./saige-finetuned`)

**Memory Requirements:**
- TinyLlama (1.1B): ~8GB VRAM
- TinyLlama (1.1B) + 4-bit: ~4GB VRAM
- Mistral (7B): ~28GB VRAM
- Mistral (7B) + 4-bit: ~8GB VRAM

**Output:**
Creates a directory with:
- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `tokenizer_config.json`, `tokenizer.json` - Tokenizer files
- `training_args.bin` - Training arguments

---

### 3. `train_pipeline.sh` - Complete Training Pipeline

Orchestrates the entire workflow: experience collection → conversion → preparation.

**Usage:**
```bash
# Default settings
./train_pipeline.sh

# Custom settings
TRAINING_EPISODES=100 \
MAX_HARM=0.2 \
MIN_COMPOSITE=6.0 \
MIN_CALIBRATION=3.0 \
./train_pipeline.sh
```

**Environment Variables:**
- `DB_PATH`: Database path (default: `../saige.db`)
- `WORKER_URL`: SAIGE worker URL (default: `http://localhost:8787`)
- `TRAINING_EPISODES`: Number of episodes to collect (default: `50`)
- `MAX_HARM`: Maximum harm threshold (default: `0.25`)
- `MIN_COMPOSITE`: Minimum composite quality score (default: `5.0`)
- `MIN_CALIBRATION`: Minimum calibration score (default: `2.0`)
- `FORMAT`: Output format (default: `mistral`)

---

## Training Formats

### ChatML / TinyLlama Format (`--format chatml`)
```
<|system|>
You are a helpful AI assistant...</s>
<|user|>
Question</s>
<|assistant|>
Answer</s>
```

### Mistral Format (`--format mistral`)
```
<s>[INST] Question [/INST] Answer</s>
```

### Llama 3 Format (`--format llama3`)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
System prompt<|eot_id|>
<|start_header_id|>user<|end_header_id|>
User message<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assistant response<|eot_id|>
```

### Alpaca Format (`--format alpaca`)
```
### Instruction:
Question

### Response:
Answer
```

---

## Training Workflows

### Workflow A: Local Training (GPU Available)

```bash
# 1. Collect and convert data
./train_pipeline.sh

# 2. Fine-tune locally
python train_local.py --data saige_training_data_v2.csv --use-4bit

# 3. Test the model
python test_model.py --model ./saige-finetuned

# 4. Deploy to SAIGE worker
# (Copy model files and update worker configuration)
```

### Workflow B: Cloud Training (Google Colab)

```bash
# 1. Generate training data
python saige_to_sft_v2.py --db ../saige.db --include-gold
```
```python
from google.colab import files
files.upload()  # Upload saige_training_data_v2.csv

# Install dependencies
!pip install transformers trl peft accelerate

# Run training
!python train_local.py --data saige_training_data_v2.csv --use-4bit
```
```bash
# 4. Download trained model from Colab
# 5. Deploy to SAIGE
```

### Workflow C: Cloudflare Workers AI + LoRA

```bash
# 1. Generate training data
python saige_to_sft_v2.py --db ../saige.db --format mistral --include-gold

# 2. Train LoRA adapter (use Colab or HuggingFace AutoTrain)

# 3. Deploy to Cloudflare Workers AI
cd ../worker
wrangler ai finetune create \
    @cf/mistral/mistral-7b-instruct-v0.2-lora \
    saige-ethics-lora \
    ./lora-adapters/

# 4. Update worker.ts to use the LoRA
# (See ../Buddhist Reference Archive/buddhist_worker_lora.js for example)
```

---

## Filtering Strategy

The converter filters experiences using **composite quality scoring** across harm, Buddhist alignment, response calibration, and coherence.

### Default Filters (Balanced)
```bash
python saige_to_sft_v2.py \
    --max-harm 0.25 \        # Max 25% harm
    --min-composite 5.0 \    # Composite score ≥ 5/10
    --min-calibration 2.0    # Calibration score ≥ 2/10
```

### Strict Filters (High Quality)
```bash
python saige_to_sft_v2.py \
    --max-harm 0.15 \        # Max 15% harm
    --min-composite 7.0 \    # Composite score ≥ 7/10
    --min-calibration 5.0    # Well-calibrated responses only
```

### Permissive Filters (More Data)
```bash
python saige_to_sft_v2.py \
    --max-harm 0.4 \         # Allow up to 40% harm
    --min-composite 3.0 \    # Composite score ≥ 3/10
    --min-calibration 1.0    # Minimal calibration floor
```

**Trade-offs:**
- **Strict**: Higher quality, fewer examples (may underfit)
- **Balanced**: Good balance of quality and quantity
- **Permissive**: More examples, lower average quality (may learn sub-optimal patterns)

---

## Troubleshooting

### "No experiences found matching criteria"

**Problem**: Not enough data or filters too strict.

**Solutions:**
1. Collect more experiences: `node trainer.js 100`
2. Relax filters: `--max-harm 0.4 --min-composite 3.0`
3. Check database: `sqlite3 ../saige.db "SELECT COUNT(*) FROM experiences"`

### "CUDA out of memory"

**Problem**: Model too large for GPU.

**Solutions:**
1. Use 4-bit quantization: `--use-4bit`
2. Reduce batch size: `--batch-size 1`
3. Use smaller model: `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
4. Use Google Colab with T4 or A100 GPU

### "Training loss not decreasing"

**Problem**: Learning rate or configuration issue.

**Solutions:**
1. Increase learning rate: `--learning-rate 5e-4`
2. Increase LoRA rank: `--lora-rank 32`
3. Train for more epochs: `--epochs 5`
4. Check data quality: run `--diagnostics-only` to inspect scores

---

## Next Steps

After training:

1. **Evaluate the model** with Buddhist principle test suite (Priority 3)
2. **Deploy to SAIGE worker** to collect new experiences with the improved model
3. **Iterate**: The new model will generate better responses → more high-quality data → better next model

This creates a **continuous improvement loop**:
```
Better Model → Better Responses → Better Training Data → Better Model
```

---

## Files

- `saige_to_sft_v2.py` - Experience to SFT converter with cleaning, calibration, and coherence scoring
- `train_local.py` - Local fine-tuning script (LoRA/QLoRA)
- `train_pipeline.sh` - Complete pipeline orchestrator
- `requirements.txt` - Python dependencies
- `trainer.js` - Experience collection

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 SAIGE Training Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Experience Collection (trainer.js)                     │
│     └─> Worker API → Database (experiences table)          │
│                                                             │
│  2. Cleaning & Scoring (saige_to_sft_v2.py)               │
│     ├─ Clean: typos, placeholders, AI prefixes            │
│     ├─ Score: calibration, coherence, Buddhist, harm      │
│     ├─ Filter: composite ≥ 5.0, harm < 0.25              │
│     └─> CSV file (text + 11 quality metadata columns)     │
│                                                             │
│  3. Format Conversion                                       │
│     ├─ ChatML: <|system|>...<|user|>...<|assistant|>     │
│     ├─ Mistral: <s>[INST]...[/INST]...</s>               │
│     └─ Llama3: <|begin_of_text|>...<|eot_id|>            │
│                                                             │
│  4. Fine-Tuning (train_local.py)                           │
│     ├─ LoRA/QLoRA parameter-efficient training            │
│     ├─ Optimized for Buddhist principles + harm reduction │
│     └─> Adapter weights (adapter_model.safetensors)       │
│                                                             │
│  5. Deployment                                              │
│     ├─ Local: Load adapter with base model                │
│     ├─ Cloudflare: Deploy LoRA to Workers AI              │
│     └─> Improved SAIGE model                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
