#!/bin/bash
# train_pipeline.sh
# Complete training pipeline for SAIGE: Experience collection → SFT conversion → Fine-tuning

set -e  # Exit on error

echo "🧘 SAIGE Training Pipeline"
echo "=========================================="
echo ""

# Configuration
DB_PATH="${DB_PATH:-../saige.db}"
WORKER_URL="${WORKER_URL:-http://localhost:8787}"
TRAINING_EPISODES="${TRAINING_EPISODES:-50}"
MAX_HARM="${MAX_HARM:-0.25}"
MIN_COMPOSITE="${MIN_COMPOSITE:-5.0}"
MIN_CALIBRATION="${MIN_CALIBRATION:-2.0}"
FORMAT="${FORMAT:-mistral}"

echo "Configuration:"
echo "  Database: $DB_PATH"
echo "  Worker URL: $WORKER_URL"
echo "  Training Episodes: $TRAINING_EPISODES"
echo "  Max Harm Threshold: $MAX_HARM"
echo "  Min Composite Score: $MIN_COMPOSITE"
echo "  Min Calibration Score: $MIN_CALIBRATION"
echo "  Output Format: $FORMAT"
echo ""

# Step 1: Collect training experiences
echo "📚 Step 1: Collecting training experiences..."
echo "   Running $TRAINING_EPISODES training episodes to generate data"
echo ""

if command -v node &> /dev/null; then
    node trainer.js $TRAINING_EPISODES
else
    echo "   ⚠️  Node.js not found, skipping experience collection"
    echo "   Make sure you have experiences in the database already"
fi

echo ""
echo "✅ Experience collection complete"
echo ""

# Step 2: Convert experiences to SFT format
echo "📝 Step 2: Converting experiences to SFT training data..."
echo ""

python3 saige_to_sft_v2.py \
    --db "$DB_PATH" \
    --output "saige_training_data_v2.csv" \
    --format "$FORMAT" \
    --max-harm "$MAX_HARM" \
    --min-composite "$MIN_COMPOSITE" \
    --min-calibration "$MIN_CALIBRATION" \
    --include-gold

if [ ! -f "saige_training_data_v2.csv" ]; then
    echo "❌ Failed to generate training data"
    exit 1
fi

echo ""
echo "✅ Training data conversion complete"
echo ""

# Step 3: Prepare for fine-tuning
echo "📦 Step 3: Preparing for fine-tuning..."
echo ""

# Count lines in training data
line_count=$(wc -l < saige_training_data_v2.csv)
example_count=$((line_count - 1))  # Subtract header

echo "   Training examples: $example_count"
echo ""

if [ $example_count -lt 10 ]; then
    echo "⚠️  Warning: Only $example_count training examples found"
    echo "   Recommendation: Collect more experiences (at least 100+)"
    echo "   Run: ./train_pipeline.sh"
    echo ""
fi

# Generate training recommendations
echo "💡 Fine-tuning recommendations:"
echo ""

if [ $example_count -lt 50 ]; then
    echo "   Dataset size: Small ($example_count examples)"
    echo "   - Epochs: 5-10"
    echo "   - Learning rate: 5e-5"
    echo "   - Batch size: 2-4"
    echo "   - LoRA rank: 8-16"
elif [ $example_count -lt 200 ]; then
    echo "   Dataset size: Medium ($example_count examples)"
    echo "   - Epochs: 3-5"
    echo "   - Learning rate: 3e-5"
    echo "   - Batch size: 4-8"
    echo "   - LoRA rank: 16-32"
else
    echo "   Dataset size: Large ($example_count examples)"
    echo "   - Epochs: 2-3"
    echo "   - Learning rate: 2e-5"
    echo "   - Batch size: 8-16"
    echo "   - LoRA rank: 32-64"
fi

echo ""
echo "=========================================="
echo "🎉 Pipeline Complete!"
echo ""
echo "Next steps:"
echo ""
echo "Option A: Local Fine-tuning (if you have GPU)"
echo "   1. Install training dependencies:"
echo "      pip install transformers torch datasets trl peft accelerate"
echo ""
echo "   2. Run local fine-tuning:"
echo "      python train_local.py --data saige_training_data_v2.csv"
echo ""
echo "Option B: Cloud Fine-tuning (Google Colab)"
echo "   1. Upload saige_training_data.csv to Google Colab"
echo "   2. Use HuggingFace AutoTrain or TRL's SFTTrainer"
echo "   3. Download the trained adapter files"
echo ""
echo "Option C: Cloudflare Workers AI with LoRA"
echo "   1. Upload saige_training_data.csv to HuggingFace"
echo "   2. Train LoRA adapter with AutoTrain"
echo "   3. Deploy to Cloudflare Workers AI:"
echo "      wrangler ai finetune create @cf/mistral/mistral-7b-instruct-v0.2-lora \\"
echo "        saige-ethics-lora ./lora-adapters/"
echo ""
echo "Training data: $(pwd)/saige_training_data_v2.csv"
echo "=========================================="
