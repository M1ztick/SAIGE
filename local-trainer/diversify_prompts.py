"""
Diversify system prompt conditions across DPO pairs.

Stratifies pairs by record_id so each record gets roughly equal coverage of:
  - "rs"      : the Right Speech system prompt (original)
  - "generic" : "You are a helpful AI assistant."
  - "none"    : no system message at all

This teaches the model that preferred behavior is unconditional, not
triggered by a specific prompt being present.

Usage:
  python3 local-trainer/diversify_prompts.py
  python3 local-trainer/diversify_prompts.py --min-delta 2   # filter weak pairs first
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

JSONL_IN  = Path(__file__).parent / "dpo_pairs.jsonl"
JSONL_OUT = Path(__file__).parent / "dpo_pairs_diversified.jsonl"

RS_PROMPT = (
    "You are a compassionate AI assistant trained in the Buddhist ethical "
    "principles of Right Speech.\n\n"
    "Your responses should be:\n"
    "- Truthful: never fabricate or speculate without clearly flagging it\n"
    "- Beneficial: optimize for what actually helps this person, not just surface accuracy\n"
    "- Timely: calibrate directness and depth to what this moment calls for\n"
    "- Non-divisive: do not frame people or groups against each other\n"
    "- Non-harsh: be firm when necessary, never contemptuous or dismissive\n"
    "- Concise: say what needs to be said; do not fill space with empty words\n\n"
    "When someone is distressed, acknowledge their situation before offering solutions."
)

GENERIC_PROMPT = "You are a helpful AI assistant."

CONDITIONS = ["rs", "generic", "none"]


def apply_condition(pair: dict, condition: str) -> dict:
    """Return a copy of pair with the system message replaced per condition."""
    pair = pair.copy()
    prompt = [m for m in pair["prompt"] if m["role"] != "system"]
    if condition == "rs":
        prompt = [{"role": "system", "content": RS_PROMPT}] + prompt
    elif condition == "generic":
        prompt = [{"role": "system", "content": GENERIC_PROMPT}] + prompt
    pair["prompt"] = prompt
    pair["prompt_condition"] = condition
    return pair


def main():
    """Diversify system prompt conditions and optionally filter by score delta."""
    delta_help = (
        "Minimum score_delta to include (default: 1.0). "
        "Note: 75%% of current pairs are at delta=1 (integers only), "
        "so delta>=2 yields only ~21 pairs — too few at this dataset size. "
        "Apply this filter only when the dataset exceeds ~300 pairs."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-delta", type=float, default=1.0, help=delta_help)
    args = parser.parse_args()

    pairs = [json.loads(l) for l in JSONL_IN.read_text().splitlines() if l.strip()]

    if args.min_delta > 1.0:
        before = len(pairs)
        pairs = [p for p in pairs if p.get("score_delta", 0) >= args.min_delta]
        print(f"Delta filter (≥{args.min_delta}): {before} → {len(pairs)} pairs")

    # Group by record_id, preserving original order within each record
    by_record: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_record[pair["record_id"]].append(pair)

    output = []
    condition_counts: dict[str, int] = defaultdict(int)
    record_summary: dict[str, dict] = {}

    for record_id, record_pairs in sorted(by_record.items()):
        record_dist: dict[str, int] = defaultdict(int)
        for i, pair in enumerate(record_pairs):
            condition = CONDITIONS[i % 3]
            output.append(apply_condition(pair, condition))
            condition_counts[condition] += 1
            record_dist[condition] += 1
        record_summary[record_id] = dict(record_dist)

    JSONL_OUT.write_text("\n".join(json.dumps(p) for p in output))

    print(f"\nInput:  {JSONL_IN.name} ({len(pairs)} pairs)")
    print(f"Output: {JSONL_OUT.name} ({len(output)} pairs)")
    print("\nGlobal condition distribution:")
    for c in CONDITIONS:
        print(f"  {c:>8}: {condition_counts[c]} pairs")
    print("\nPer-record breakdown:")
    for record_id, dist in record_summary.items():
        row = "  ".join(f"{c}={dist.get(c, 0)}" for c in CONDITIONS)
        print(f"  {record_id}: {row}")


if __name__ == "__main__":
    main()
