#!/usr/bin/env python3
"""
judge_rubric.py — Rubric-based LLM judge for SAIGE ablation results.

Scores every generation in an ablation_results.json (produced by
SAIGE_DPO_v3_Inference.ipynb) against the evaluation_questions of the
annotation records each eval prompt is tagged with. Reuses the anchored
judge from generate_dpo_pairs.py (score_candidate), so eval scores share
the calibration of the chosen/rejected scores in the training data.

For each scenario x condition x record the judge returns per-question
scores, a flaw list, and a reconciled 0-10 overall. The report aggregates
per condition and prints the three comparisons that matter:

  base+RS   - base+gen    what the system prompt alone buys on the base model
  SAIGE+RS  - base+RS     adapter effect under the trained prompt
  SAIGE+gen - base+gen    adapter effect with no RS prompt — behavior in the
                          weights; this is the headline number

Usage:
  python judge_rubric.py ablation_results.json
  python judge_rubric.py ablation_results.json --out judge_scores.json
  python judge_rubric.py ablation_results.json --dry-run   # plumbing check, no API

Requires ANTHROPIC_API_KEY in the environment (same as generate_dpo_pairs.py).
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

from generate_dpo_pairs import (
    ANNOTATIONS_DIR,
    JUDGE_PROMPT_TAG,
    _cache_key,
    load_annotation_records,
    score_candidate,
)

CONDITIONS = ["base+RS", "base+gen", "SAIGE+RS", "SAIGE+gen"]
CACHE_FILE = Path(__file__).parent / ".judge_rubric_cache.json"

# Matches the --judge-model default in generate_dpo_pairs.py so eval overalls are
# directly comparable to the chosen/rejected scores in dpo_pairs.jsonl. Note the
# shared _call_api helper passes a temperature, which claude-opus-5-generation
# models reject — stay on a model that accepts it unless _call_api changes too.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def judge_all(scenarios: list, records: dict, judge_model: str,
              use_cache: bool, dry_run: bool) -> list:
    """Score every (scenario, condition, record) triple; returns flat row list."""
    client = None
    if not dry_run:
        import anthropic  # deferred so --dry-run works without the SDK installed
        client = anthropic.Anthropic()

    cache = _load_cache() if use_cache else {}
    rows = []
    total = sum(
        len([c for c in CONDITIONS if sc["outputs"].get(c)]) * len(sc["record_ids"])
        for sc in scenarios
    )
    done = 0

    for sc in scenarios:
        for cond in CONDITIONS:
            text = sc["outputs"].get(cond)
            if not text:
                continue
            for rid in sc["record_ids"]:
                record = records.get(rid)
                if record is None:
                    print(f"  WARNING: no annotation record {rid} — skipping", file=sys.stderr)
                    continue
                done += 1
                key = _cache_key(rid, JUDGE_PROMPT_TAG, judge_model, cond,
                                 sc["user_message"], text)
                if dry_run:
                    ev = {"flaws": [], "scores": [], "overall": 5,
                          "overall_judge": 5, "summary": "dry run"}
                elif key in cache:
                    ev = cache[key]
                else:
                    ev = score_candidate(client, sc["user_message"], text, record, judge_model)
                    cache[key] = ev
                    if use_cache:
                        _save_cache(cache)
                print(f"  [{done}/{total}] {sc['label']} | {cond} | {rid}"
                      f" -> {ev.get('overall', '?')}")
                rows.append({
                    "label": sc["label"],
                    "condition": cond,
                    "record_id": rid,
                    "truncated": sc.get("truncated", {}).get(cond, False),
                    "overall": ev.get("overall", 5),
                    "overall_judge": ev.get("overall_judge", ev.get("overall", 5)),
                    "flaws": ev.get("flaws", []),
                    "scores": ev.get("scores", []),
                    "summary": ev.get("summary", ""),
                })
    return rows


def aggregate(rows: list) -> dict:
    """Per-condition means plus the three deltas the ablation exists to measure."""
    by_cond = {
        c: [r["overall"] for r in rows if r["condition"] == c] for c in CONDITIONS
    }
    means = {c: _mean(v) for c, v in by_cond.items()}
    return {
        "n_per_condition": {c: len(v) for c, v in by_cond.items()},
        "mean_overall": means,
        "comparisons": {
            "prompt_effect_on_base (base+RS - base+gen)":
                means["base+RS"] - means["base+gen"],
            "adapter_effect_under_RS (SAIGE+RS - base+RS)":
                means["SAIGE+RS"] - means["base+RS"],
            "behavior_in_weights (SAIGE+gen - base+gen)":
                means["SAIGE+gen"] - means["base+gen"],
        },
    }


def print_report(rows: list, agg: dict) -> None:
    print(f"\n{'=' * 68}\nPER-SCENARIO OVERALL (mean across records)\n{'=' * 68}")
    labels = list(dict.fromkeys(r["label"] for r in rows))
    header = f"{'scenario':<42}" + "".join(f"{c:>10}" for c in CONDITIONS)
    print(header)
    for label in labels:
        cells = []
        for cond in CONDITIONS:
            vals = [r["overall"] for r in rows
                    if r["label"] == label and r["condition"] == cond]
            cells.append(f"{_mean(vals):>10.1f}" if vals else f"{'—':>10}")
        print(f"{label[:41]:<42}" + "".join(cells))

    print(f"\n{'=' * 68}\nAGGREGATE\n{'=' * 68}")
    for cond in CONDITIONS:
        print(f"  {cond:<10} mean overall {agg['mean_overall'][cond]:.2f}  "
              f"(n={agg['n_per_condition'][cond]})")
    print("\n  Comparisons (positive = improvement):")
    for name, delta in agg["comparisons"].items():
        print(f"    {name:<52} {delta:+.2f}")

    n_trunc = sum(1 for r in rows if r["truncated"])
    if n_trunc:
        print(f"\n  NOTE: {n_trunc} scored generations hit the length cap — "
              "treat length-related sub-scores (rs-012) with caution.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score ablation generations against annotation-record rubrics."
    )
    parser.add_argument("results", help="ablation_results.json from the inference notebook")
    parser.add_argument("--out", default="judge_scores.json",
                        help="Where to write per-row scores + aggregate (default: judge_scores.json)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL}, the same judge "
                             "that scored the training pairs — keeps scores comparable)")
    parser.add_argument("--annotations-dir", default=str(ANNOTATIONS_DIR),
                        help="Path to annotation records directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached judge results and rescore everything")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the plumbing with placeholder scores, no API calls")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(f"Loaded {len(scenarios)} scenarios from {args.results}")

    print("Loading annotation records:")
    records = {
        r["id"]: r
        for r in load_annotation_records(
            Path(args.annotations_dir), ["draft", "committed", "final"]
        )
    }

    rows = judge_all(scenarios, records, args.judge_model,
                     use_cache=not args.no_cache, dry_run=args.dry_run)
    if not rows:
        sys.exit("No generations scored — check the results file and record IDs.")

    agg = aggregate(rows)
    print_report(rows, agg)

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"judge_model": args.judge_model, "rows": rows, "aggregate": agg},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
