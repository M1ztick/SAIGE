"""SAIGE ablation v3 — corrected logprob-margin evaluation of DPO adapters.

WHAT CHANGED VS v2 (and why v2's absolute numbers are wrong)
-----------------------------------------------------------
v2's `score_pair` built the scored completion with

    tokenizer.apply_chat_template(completion_msgs, tokenize=False,
                                  add_generation_prompt=False)

On Qwen2.5's template, rendering an assistant-only message list emits the
template's *default system turn* plus a fresh assistant header:

    '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a
     helpful assistant.<|im_end|>\n<|im_start|>assistant\nThat sounds
     draining.<|im_end|>\n'

Appended after a prompt that already ended in '<|im_start|>assistant\n', this
scores the response tokens after a malformed mid-conversation second system
turn, and pads every completion with ~24 junk tokens (measured on a short
reply; the junk block is a constant 24 tokens). Both the total-nat margins
and the per-token margins in ablation_results_v2.json are therefore wrong in
absolute terms. Deltas vs base are largely spared because the junk block is
identical across states, which is why v2 and the earlier run5 script agreed on
deltas while disagreeing by up to 16 nats per record on absolute values.

v3 scores exactly the assistant content plus the '<|im_end|>\n' turn
terminator, which is what the model actually generates.

STATISTICS
----------
v2 reported the mean of per-record total-nat margins. Those margins range from
-150 to +130 and are dominated by response length and style, so the mean is
noise-dominated and preference accuracy is pinned by the base model's prior
rather than by the adapter. v3 reports the *paired* per-record delta vs base
(the same record, the same condition, adapter minus base), which cancels the
per-record offset, plus an exact two-sided sign test and a bootstrap CI.

EVAL SETS
---------
--eval-set legacy   : the 11 grouped-split records of the OLD dataset
                      (M1ztyk/SAIGE-right-speech-dpo, seed 42). These pairs
                      are "good vs slightly better" (mean judge delta 0.42
                      under the v4 judge), so they cannot show whether an
                      adapter trained on high-contrast pairs learned anything.
                      Kept for continuity with the v1/v2 tables only.
--eval-set heldout  : high-contrast held-out pairs (delta >= 3) produced by the
                      scenario-grouped split in generate_rejects_v5.py. This is
                      the eval set that actually matches the v5+ training
                      distribution. Use this one for conclusions.
--eval-set both     : score both, report separately (default).

A NOTE ON THE "none" CONDITION
------------------------------
Qwen2.5's chat template injects its own default system turn when no system
message is supplied, so "none" is really "Qwen's default system prompt", not
"no system prompt". v3 keeps the key `none` for table continuity and records
the true meaning in `condition_notes` in the output JSON.

USAGE
    ABLATION_DRY_RUN=1 python ablation_v3.py    # CPU, no model: validates data
                                                # prep + templating + prefix
    python ablation_v3.py                       # full scoring (needs GPU)
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

OLD_REPO = "M1ztyk/SAIGE-right-speech-dpo"
OLD_FILE = "dpo_pairs_diversified.jsonl"
V4_REPO = "M1ztyk/SAIGE-right-speech-dpo-v4"
HELDOUT_FILE = "dpo_pairs_v5_eval.jsonl"   # written by generate_rejects_v5.py
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT_REPO = "M1ztyk/saige-2x2-ablation-results"
OUT_FILE = "ablation_results_v3.json"

ADAPTERS = {
    "base": None,
    "run3": "M1ztyk/SAIGE-dpo-v3-run3",
    "run4": "M1ztyk/SAIGE-dpo-v3-run4",
    "run5": "M1ztyk/SAIGE-dpo-v5",
}

# NOTE (v5 fix): RS_PROMPT is now byte-identical to SAIGE_SYSTEM_PROMPT, the
# prompt the CHOSEN responses are actually generated under. Through v4 the two
# differed by a single trailing newline, which put one extra token inside the
# system turn and meant the "rs" evaluation condition was never the same string
# the chosen text was produced under. One token is small, but this is an
# experiment about prompt conditioning, so the confound is removed rather than
# tolerated. Absolute rs-condition numbers therefore shift slightly against the
# v1/v2 tables — which are being discarded as unsound anyway (see ablation_v3).
RS_PROMPT = """\
You are a compassionate AI assistant trained in Buddhist ethical principles of Right Speech.

Your responses should be:
- Truthful: never fabricate or speculate without clearly flagging it
- Beneficial: optimize for what actually helps this person, not just surface accuracy
- Timely: calibrate directness and depth to what this moment calls for
- Non-divisive: do not frame people or groups against each other
- Non-harsh: be firm when necessary, never contemptuous or dismissive
- Concise: say what needs to be said; do not fill space with empty words

When someone is distressed, acknowledge their situation before offering solutions.\
"""

GENERIC_PROMPT = "You are a helpful AI assistant."

CONDITION_PROMPTS = {"rs": RS_PROMPT, "generic": GENERIC_PROMPT, "none": None}

CONDITION_NOTES = {
    "rs": "canonical SAIGE Right Speech system prompt",
    "generic": "'You are a helpful AI assistant.'",
    "none": (
        "NO system message is supplied, but Qwen2.5's chat template then injects "
        "its own default system turn ('You are Qwen, created by Alibaba Cloud. You "
        "are a helpful assistant.'). This condition is therefore 'Qwen default "
        "system prompt', NOT 'no system prompt'."
    ),
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- statistics
def sign_test_p(n_pos, n_total):
    """Exact two-sided binomial sign test at p=0.5. Ties excluded by caller."""
    if n_total == 0:
        return None
    k = min(n_pos, n_total - n_pos)
    tail = sum(math.comb(n_total, i) for i in range(0, k + 1)) / (2 ** n_total)
    return min(1.0, 2 * tail)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean."""
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def paired_stats(deltas):
    """Summarize paired per-record deltas (adapter - base)."""
    nonzero = [d for d in deltas if d != 0]
    n_pos = sum(1 for d in nonzero if d > 0)
    lo, hi = bootstrap_ci(deltas)
    return {
        "n": len(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
        "n_positive": n_pos,
        "n_nonzero": len(nonzero),
        "sign_test_p": sign_test_p(n_pos, len(nonzero)),
        "bootstrap_ci95": [lo, hi],
        "per_record_deltas": deltas,
    }


# ---------------------------------------------------------------- data
def _records_from_rows(rows, label):
    """One record per unique chosen text; all 3 conditions synthesized."""
    records, seen = [], set()
    for row in rows:
        key = json.dumps(row["chosen"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        user_msgs = [m for m in row["prompt"] if m["role"] == "user"]
        assert user_msgs, f"{label}: eval record has no user message"
        records.append(
            {
                "user": user_msgs,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "score_delta": row.get("score_delta"),
                "record_id": row.get("record_id"),
            }
        )
    return records


def build_legacy_records():
    """Reproduce run4's grouped eval split (17 rows / 11 records, seed 42)."""
    from datasets import load_dataset

    ds = load_dataset(OLD_REPO, data_files=OLD_FILE, split="train")
    keep = [c for c in ("prompt", "chosen", "rejected", "record_id") if c in ds.column_names]
    ds = ds.select_columns(keep)
    assert len(ds) == 85, f"Expected 85 diversified pairs, found {len(ds)}"

    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[json.dumps(row["chosen"], sort_keys=True)].append(i)
    group_keys = sorted(groups)
    random.Random(42).shuffle(group_keys)
    eval_idx = []
    for key in group_keys:
        if len(eval_idx) >= round(0.2 * len(ds)):
            break
        eval_idx.extend(groups[key])
    eval_ds = ds.select(sorted(set(eval_idx)))
    assert len(eval_ds) == 17, f"Grouped split should yield 17 rows, got {len(eval_ds)}"

    records = _records_from_rows(eval_ds, "legacy")
    assert len(records) == 11, f"Expected 11 unique legacy records, got {len(records)}"
    log(f"legacy eval records: {len(records)} (17 rows, grouped split seed 42)")
    return records


def build_heldout_records():
    """High-contrast held-out pairs from the v5 generation split."""
    from datasets import load_dataset

    local_dir = os.environ.get("SAIGE_LOCAL_DATA", "")
    try:
        if local_dir:
            path = os.path.join(local_dir, HELDOUT_FILE)
            assert os.path.exists(path), f"SAIGE_LOCAL_DATA set but {path} missing"
            ds = load_dataset("json", data_files=path, split="train")
        else:
            ds = load_dataset(V4_REPO, data_files=HELDOUT_FILE, split="train")
    except Exception as e:  # noqa: BLE001
        log(f"heldout eval set unavailable ({type(e).__name__}: {e})")
        log(f"  -> run generate_rejects_v5.py first; it writes {HELDOUT_FILE}")
        return None
    keep = [c for c in ("prompt", "chosen", "rejected", "score_delta", "record_id",
                        "eval_stratum")
            if c in ds.column_names]
    ds = ds.select_columns(keep)
    # Dedupe on (chosen, rejected), NOT chosen alone: the held-out set carries two
    # strata per prompt (same chosen, different persona-generated rejected), and
    # chosen-only dedup would silently discard the unseen-persona stratum.
    records, seen = [], set()
    for row in ds:
        key = (json.dumps(row["chosen"], sort_keys=True),
               json.dumps(row["rejected"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        user_msgs = [m for m in row["prompt"] if m["role"] == "user"]
        assert user_msgs, "heldout: eval record has no user message"
        records.append({
            "user": user_msgs,
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "score_delta": row.get("score_delta"),
            "record_id": row.get("record_id"),
            "eval_stratum": row.get("eval_stratum"),
        })
    deltas = [r["score_delta"] for r in records if r["score_delta"] is not None]
    strata = defaultdict(int)
    for r in records:
        strata[r.get("eval_stratum")] += 1
    log(f"heldout eval records: {len(records)}"
        + (f" | judge score_delta mean {sum(deltas)/len(deltas):.2f}" if deltas else "")
        + (f" | strata {dict(strata)}" if strata else ""))
    if deltas and min(deltas) < 3:
        log(f"WARNING: heldout set contains a pair at delta {min(deltas)} (<3) — "
            f"contrast no longer matches the training distribution")
    return records


# ---------------------------------------------------------------- scoring
def build_prompt_text(tokenizer, sys_prompt, user_msgs):
    msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + user_msgs
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_completion_text(completion_msgs):
    """The assistant turn EXACTLY as the model emits it.

    Deliberately NOT apply_chat_template: on Qwen2.5 that re-emits a default
    system turn and a second assistant header (the v2 bug). The prompt already
    ends with '<|im_start|>assistant\\n', so the completion is the content plus
    the turn terminator.
    """
    content = "\n".join(m["content"] for m in completion_msgs if m["role"] == "assistant")
    return content + "<|im_end|>\n"


def score_completion(model, tokenizer, prompt_text, completion_text):
    """Return (total_logp, n_completion_tokens), or None on a BPE boundary merge."""
    import torch

    ids_prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ids_full = tokenizer(prompt_text + completion_text, add_special_tokens=False)["input_ids"]
    if ids_full[: len(ids_prompt)] != ids_prompt:
        return None
    comp_start = len(ids_prompt)

    input_ids = torch.tensor([ids_full], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    token_lp = logprobs[torch.arange(len(targets)), targets]
    return token_lp[comp_start - 1:].sum().item(), len(ids_full) - comp_start


def load_model_and_tokenizer(adapter_id):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 (CC 7.5): fp16, not bf16
        bnb_4bit_use_double_quant=True,
    )
    # `dtype` is the current kwarg; transformers <4.54 only accepts `torch_dtype`.
    # Colab ships whatever it ships, so accept either rather than pinning.
    kw = dict(quantization_config=bnb, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, **kw)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_id is not None:
        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    log(f"  loaded state: {adapter_id or 'base'}")
    return model, tokenizer


def score_state(state, adapter_id, records):
    """Return {condition: {per_record_margins, per_record_margins_per_token, ...}}."""
    model, tokenizer = load_model_and_tokenizer(adapter_id)
    out = {}
    for cond, sys_prompt in CONDITION_PROMPTS.items():
        margins, per_tok, dropped = [], [], 0
        for rec in records:
            prompt_text = build_prompt_text(tokenizer, sys_prompt, rec["user"])
            cells = {}
            for name, msgs in (("chosen", rec["chosen"]), ("rejected", rec["rejected"])):
                r = score_completion(model, tokenizer, prompt_text, build_completion_text(msgs))
                if r is None:
                    break
                cells[name] = r
            if len(cells) != 2:
                dropped += 1
                margins.append(None)
                per_tok.append(None)
                continue
            lp_c, n_c = cells["chosen"]
            lp_r, n_r = cells["rejected"]
            margins.append(lp_c - lp_r)
            per_tok.append(lp_c / max(n_c, 1) - lp_r / max(n_r, 1))
        ok = [m for m in margins if m is not None]
        out[cond] = {
            "per_record_margins": margins,
            "per_record_margins_per_token": per_tok,
            "mean_margin": sum(ok) / len(ok) if ok else None,
            "preference_accuracy": (sum(m > 0 for m in ok) / len(ok)) if ok else None,
            "n_scored": len(ok),
            "n_dropped_bpe_merge": dropped,
        }
        log(f"  {state:5s} | {cond:7s} | mean margin {out[cond]['mean_margin']:+8.2f} nats "
            f"| acc {out[cond]['preference_accuracy']:.2f} | n={len(ok)}"
            + (f" | dropped {dropped}" if dropped else ""))
    del model
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return out


def score_eval_set(name, records):
    raw = {}
    for state, adapter_id in ADAPTERS.items():
        log(f"Scoring state: {state} [{name}]")
        raw[state] = score_state(state, adapter_id, records)

    # paired deltas vs base, per condition
    paired = {}
    for state in ADAPTERS:
        if state == "base":
            continue
        paired[state] = {}
        for cond in CONDITION_PROMPTS:
            a = raw[state][cond]["per_record_margins"]
            b = raw["base"][cond]["per_record_margins"]
            deltas = [x - y for x, y in zip(a, b) if x is not None and y is not None]
            st = paired_stats(deltas)
            paired[state][cond] = st
            log(f"  PAIRED {state:5s} | {cond:7s} | mean d {st['mean_delta']:+7.3f} "
                f"| {st['n_positive']}/{st['n_nonzero']} positive "
                f"| sign p {st['sign_test_p']:.4f} "
                f"| CI95 [{st['bootstrap_ci95'][0]:+.2f}, {st['bootstrap_ci95'][1]:+.2f}]")

    # per-stratum paired deltas (heldout set only: unseen scenario vs
    # unseen scenario AND unseen persona). Reported separately because
    # generalizing to a new scenario is a weaker claim than generalizing to a
    # failure mode the adapter was never trained against.
    strata = sorted({r.get("eval_stratum") for r in records if r.get("eval_stratum")})
    by_stratum = {}
    if strata:
        for stratum in strata:
            idx = [i for i, r in enumerate(records) if r.get("eval_stratum") == stratum]
            by_stratum[stratum] = {}
            for state in ADAPTERS:
                if state == "base":
                    continue
                by_stratum[stratum][state] = {}
                for cond in CONDITION_PROMPTS:
                    a = raw[state][cond]["per_record_margins"]
                    b = raw["base"][cond]["per_record_margins"]
                    deltas = [a[i] - b[i] for i in idx
                              if a[i] is not None and b[i] is not None]
                    if deltas:
                        by_stratum[stratum][state][cond] = paired_stats(deltas)
            log(f"  stratum {stratum}: n={len(idx)}")

    # rs/generic gap per state (prompt-sensitivity check)
    gaps = {
        state: (raw[state]["rs"]["mean_margin"] - raw[state]["generic"]["mean_margin"])
        for state in ADAPTERS
        if raw[state]["rs"]["mean_margin"] is not None
    }
    return {
        "n_records": len(records),
        "absolute": raw,
        "paired_vs_base": paired,
        "paired_vs_base_by_stratum": by_stratum,
        "rs_minus_generic_gap": gaps,
        "gap_note": (
            "A flat rs-minus-generic gap means an adapter shifted all prompt "
            "conditions in parallel. For an adapter that did not shift anything "
            "(run5 on the legacy set), a flat gap is evidence of nothing."
        ),
    }


# ---------------------------------------------------------------- dry run
def dry_run(eval_sets):
    """CPU-only: validate templating, the v2-bug fix, and the prefix property."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 1. the v2 bug is real and this script does not reproduce it
    probe = [{"role": "assistant", "content": "That sounds draining."}]
    buggy = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    fixed = build_completion_text(probe)
    assert "You are Qwen" in buggy, "expected v2's default-system injection in the probe"
    assert "You are Qwen" not in fixed, "v3 completion text must not inject a system turn"
    assert not fixed.startswith("<|im_start|>"), "v3 completion must not open a new turn"
    junk = len(tok(buggy)["input_ids"]) - len(tok(fixed)["input_ids"])
    log(f"DRY RUN: v2 templating injected {junk} junk tokens per completion; v3 emits none")

    total, merges = 0, 0
    for name, records in eval_sets.items():
        if not records:
            continue
        for rec in records:
            for cond, sys_prompt in CONDITION_PROMPTS.items():
                prompt_text = build_prompt_text(tok, sys_prompt, rec["user"])
                assert "<|im_start|>" in prompt_text, f"bad template for {cond}"
                if cond == "none":
                    assert "You are Qwen" in prompt_text, (
                        "'none' should carry Qwen's default system turn — check CONDITION_NOTES"
                    )
                for msgs in (rec["chosen"], rec["rejected"]):
                    comp = build_completion_text(msgs)
                    ids_p = tok(prompt_text, add_special_tokens=False)["input_ids"]
                    ids_f = tok(prompt_text + comp, add_special_tokens=False)["input_ids"]
                    total += 1
                    if ids_f[: len(ids_p)] != ids_p:
                        merges += 1
        log(f"DRY RUN: {name} — {len(records)} records x 3 conditions templated cleanly")
    log(f"DRY RUN: prefix property held for {total - merges}/{total} completions "
        f"({merges} BPE boundary merges would be dropped)")
    assert merges / max(total, 1) < 0.05, f"too many BPE boundary merges: {merges}/{total}"

    # 3. statistics sanity
    assert abs(sign_test_p(11, 11) - 2 * (0.5 ** 11)) < 1e-12
    assert sign_test_p(6, 11) > 0.5
    lo, hi = bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0
    log("DRY RUN: sign test + bootstrap CI sanity checks passed")
    log("DRY RUN OK")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=["legacy", "heldout", "both"], default="both")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    eval_sets = {}
    if args.eval_set in ("legacy", "both"):
        eval_sets["legacy"] = build_legacy_records()
    if args.eval_set in ("heldout", "both"):
        h = build_heldout_records()
        if h:
            eval_sets["heldout"] = h
    assert eval_sets, "no eval set could be built"

    if os.environ.get("ABLATION_DRY_RUN", "") in ("1", "true"):
        dry_run(eval_sets)
        return

    payload = {
        "method": (
            "margin = logp(chosen|prompt) - logp(rejected|prompt), total nats. "
            "Completion = assistant content + '<|im_end|>' only (v2's "
            "apply_chat_template call injected a default system turn + a second "
            "assistant header; its absolute margins are not comparable to these). "
            "Headline statistic is the PAIRED per-record delta vs base."
        ),
        "base_model": BASE_MODEL,
        "adapters": ADAPTERS,
        "conditions": list(CONDITION_PROMPTS),
        "condition_notes": CONDITION_NOTES,
        "supersedes": ["ablation_results_v2.json", "ablation_results_run5.json",
                       "ablation_results.json"],
        "eval_sets": {},
    }
    for name, records in eval_sets.items():
        log(f"=== eval set: {name} ({len(records)} records) ===")
        payload["eval_sets"][name] = score_eval_set(name, records)

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {OUT_FILE}")

    if not args.no_push:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=OUT_FILE, path_in_repo=OUT_FILE,
            repo_id=OUT_REPO, repo_type="dataset",
        )
        log(f"Pushed {OUT_FILE} to https://huggingface.co/datasets/{OUT_REPO}")
    log("ABLATION COMPLETE")


if __name__ == "__main__":
    main()
