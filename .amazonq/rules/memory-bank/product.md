# SAIGE — Product Overview

## What SAIGE Is

SAIGE (Systems-Aware Independently-Governing Ethics) is an AI alignment framework that trains language models to reason ethically through **experiential learning** grounded in **Buddhist wisdom principles**. Rather than hard-coding rules, SAIGE teaches ethical behavior by having a model accumulate scored experiences and iteratively self-improve.

Live worker endpoint: `https://buddhist-ai-worker.mistykmedia.workers.dev`

---

## Core Value Proposition

- Ethics via experience, not rules — models learn by doing and scoring, not by following a fixed ruleset
- Two-dimensional scoring: harm avoidance (what to avoid) + Buddhist principle embodiment (what to embody)
- Continuous improvement loop: deploy → collect experiences → score/filter → fine-tune → redeploy
- Edge-native inference via Cloudflare Workers with D1 (SQLite) for experience storage

---

## Key Features

### Harm Scoring (4 dimensions)
- Deception — fabricating or misleading
- Harshness — contemptuous or dismissive tone
- Omission — leaving out critical truths
- Manipulation — exploiting emotional vulnerabilities

### Buddhist Alignment Scoring (5 principles, 0–10 each)
| Principle | Weight | Meaning |
|-----------|--------|---------|
| Ahimsa (non-harm) | 25% | Avoid causing suffering |
| Sacca (truthfulness) | 20% | Honest, accurate communication |
| Karuna (compassion) | 25% | Genuine care for the person |
| Panna (wisdom) | 20% | Contextual understanding, root-cause thinking |
| Upekkha (equanimity) | 10% | Calm, tone-matched, non-reactive |

### Composite Quality Scoring
```
composite = Buddhist×0.35 + Calibration×0.30 + Coherence×0.20 + (1−Harm)×10×0.15
```

### DPO (Direct Preference Optimization) Support
- Annotated chosen/rejected response pairs stored in `local-trainer/dpo_pairs.jsonl`
- Each pair scored with per-dimension evaluations (truthfulness, non-divisiveness, non-harshness, conciseness)
- Pairs keyed to canonical Buddhist texts (e.g., SN 45.8 — Right Speech)

---

## Target Users

- AI alignment researchers experimenting with values-based fine-tuning
- ML engineers building ethical conversational AI systems
- Developers integrating lightweight edge inference with iterative model improvement

---

## Use Cases

1. Generate SFT (Supervised Fine-Tuning) training data from accumulated experiences
2. Fine-tune small LLMs (TinyLlama 1.1B, Mistral 7B) with LoRA/QLoRA for ethical response generation
3. Run live inference at the edge (Cloudflare Worker) with real-time harm and alignment scoring
4. Build DPO training pairs with human-annotated preference scores for contrastive learning

---

## Current Dataset (as of last sync)

| Metric | Value |
|--------|-------|
| Total experiences | 150 |
| High-quality (passed filter) | 91 |
| Avg harm score | 0.055 |
| Avg Buddhist score | 7.24 |
| Alignment distribution | 5.5% excellent / 94.5% good |
| Difficulty range | Levels 1–4 |
| DPO annotated pairs | 8 (saige-rs-001 through saige-rs-008) |

---

## Roadmap Status

- [x] Buddhist principle scoring (Priority 1)
- [x] RL-to-SFT training pipeline (Priority 2)
- [x] Conversational calibration training
- [x] v2 converter with composite scoring
- [x] DPO pair generation and annotation pipeline
- [ ] Buddhist ethics evaluation suite (Priority 3)
- [ ] First fine-tuned model checkpoint
- [ ] Continuous collection → retrain loop
