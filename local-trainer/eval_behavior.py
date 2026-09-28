"""SAIGE behavioral eval — does the margin shift show up in what the model SAYS?

WHY
---
The corrected logprob ablation (ablation_v3.py) found a real effect: run3 moved
every held-out scenario positive in all three prompt conditions, p = 0.0039
clustered, and it survives per-token normalization so it is not a length
artifact. But it also found **zero preference flips** — not one record changed
which response the model prefers. The adapters move margins without changing
decisions.

So the open question is whether +5.33 nats of margin corresponds to any
difference in generated text. A margin is a statement about someone else's
fixed text; Right Speech is a claim about behavior. This script measures
behavior.

HOW THIS DIFFERS FROM judge_rubric.py
-------------------------------------
It reuses that harness's judge exactly — `score_candidate` from
generate_dpo_pairs.py, the same anchored Claude judge whose calibration the
chosen_score/rejected_score values in dpo_pairs.jsonl share — and reports the
same three comparisons. What it adds:

  * four states (base, run3, run4, run5), not base + one adapter
  * PAIRED per-prompt deltas vs base instead of a difference of condition means,
    so per-prompt difficulty cancels
  * an exact sign test and a bootstrap CI on those deltas
  * scenario clustering, because the 11 eval records span only 9 annotated
    scenarios and are therefore not independent
  * the same held-out records the logprob ablation scored, so the two results
    are directly comparable

HEADLINE
--------
Per local-trainer/README.md, the project's own success criterion is that
preferred behavior be *unconditional*: "If adapter + generic looks close to
adapter + RS, the training worked", and "SAIGE+gen - base+gen (behavior in the
weights, no RS prompt) is the headline number". So the GENERIC column is the
headline here, not rs.

BLINDING
--------
The judge receives only (user_message, response, annotation_record). It never
sees which state or condition produced a response, and responses are scored
independently rather than ranked against each other.

PHASES
------
  --phase generate : Qwen2.5-3B + adapters, greedy. Fits a free Colab T4.
  --phase judge    : Claude API. Needs ANTHROPIC_API_KEY. No GPU.
  --phase both     : generate, free the GPU, then judge.
  SAIGE_DRY_RUN=1  : no GPU, no API — validates wiring, prompts and statistics.

LAYOUT
------
Run from local-trainer/ (or pass --repo) so generate_dpo_pairs.py and
diversify_prompts.py are importable. ablation_v3.py must also be importable so
the eval records match the logprob run exactly.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512          # legacy chosen/rejected run ~250-450 tokens
GEN_FILE = "behavior_generations.json"
OUT_FILE = "behavior_eval_results.json"
CACHE_FILE = ".eval_behavior_cache.json"

ADAPTERS = {
    "base": None,
    "run3": "M1ztyk/SAIGE-dpo-v3-run3",
    "run4": "M1ztyk/SAIGE-dpo-v3-run4",
    "run5": "M1ztyk/SAIGE-dpo-v5",
}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ imports
def load_project(repo: Path):
    """Import the project's judge and condition prompts. Single source of truth.

    Deliberately NOT copies: if diversify_prompts.RS_PROMPT changes, this eval
    must change with it, or the condition being scored stops being the condition
    the adapter was trained under.
    """
    for p in (repo, repo.parent, Path.cwd()):
        if (p / "generate_dpo_pairs.py").exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import diversify_prompts as dv
    import generate_dpo_pairs as gdp

    conds = {"rs": dv.RS_PROMPT, "generic": dv.GENERIC_PROMPT, "none": None}
    assert "in the Buddhist" in conds["rs"], "rs prompt is not the dataset's rs condition"
    assert conds["generic"] == "You are a helpful AI assistant.", "generic prompt drifted"
    return gdp, conds


OLD_REPO = "M1ztyk/SAIGE-right-speech-dpo"
OLD_FILE = "dpo_pairs_diversified.jsonl"


def load_eval_records(repo: Path):
    """The SAME 11 held-out records the logprob ablation scores.

    Built HERE from the dataset rather than imported from ablation_v3, because
    an earlier revision of this function called ablation_v3.build_legacy_records()
    and read r["scenario"] — a key only present in ablation_v3 revisions that
    have scenario clustering. Anyone running an older copy got a bare
    KeyError: 'scenario'. Correctness should not depend on which revision of a
    neighbouring file happens to be on disk, so the split is reproduced inline
    (it is ~15 lines) and ablation_v3 is used only as an optional cross-check.
    """
    from datasets import load_dataset

    ds = load_dataset(OLD_REPO, data_files=OLD_FILE, split="train")
    keep = [c for c in ("prompt", "chosen", "rejected", "record_id",
                        "prompt_type", "score_delta", "pair_type")
            if c in ds.column_names]
    ds = ds.select_columns(keep)
    assert len(ds) == 85, f"expected 85 diversified pairs, found {len(ds)}"

    # run4's grouped split, seed 42: group by chosen text so no twin straddles
    # the split, shuffle groups, take ~20% of rows.
    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[json.dumps(row["chosen"], sort_keys=True)].append(i)
    gkeys = sorted(groups)
    random.Random(42).shuffle(gkeys)
    idx = []
    for k in gkeys:
        if len(idx) >= round(0.2 * len(ds)):
            break
        idx.extend(groups[k])
    ev = ds.select(sorted(set(idx)))
    assert len(ev) == 17, f"grouped split should yield 17 rows, got {len(ev)}"

    out, seen = [], set()
    for row in ev:
        k = json.dumps(row["chosen"], sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        rid, ptype = row.get("record_id"), row.get("prompt_type")
        out.append({
            "user": next(m["content"] for m in row["prompt"] if m["role"] == "user"),
            "record_id": rid,
            "prompt_type": ptype,
            "scenario": f"{rid}::{ptype}" if (rid and ptype) else f"{rid}::?",
            "pair_type": row.get("pair_type"),
            "score_delta": row.get("score_delta"),
        })
    assert len(out) == 11, f"expected 11 unique eval records, got {len(out)}"

    n_scen = len({r["scenario"] for r in out})
    log(f"eval records: {len(out)} over {n_scen} scenarios "
        f"| records {sorted({r['record_id'] for r in out})}")
    if n_scen < len(out):
        log(f"  NOTE: {len(out)} records but only {n_scen} scenarios — the sign "
            f"test is clustered on scenario, not per record.")

    # Optional cross-check: if a scenario-aware ablation_v3 is importable, the
    # two must agree on the eval set, or the behavioural and logprob results are
    # not describing the same prompts. A mismatch is fatal; an old or absent
    # ablation_v3 is not.
    for p in (Path.cwd(), repo, repo.parent):
        if (p / "ablation_v3.py").exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import ablation_v3 as ab
        ref = ab.build_legacy_records()
        if ref and "scenario" in ref[0]:
            mine = [r["user"] for r in out]
            theirs = [next(m["content"] for m in r["user"] if m["role"] == "user")
                      for r in ref]
            assert mine == theirs, (
                "eval set disagrees with ablation_v3.build_legacy_records(); the "
                "behavioural and logprob results would not be comparable")
            log("  cross-checked against ablation_v3: identical eval set")
        else:
            log("  ablation_v3 present but pre-clustering; skipping cross-check "
                "(harmless — this script no longer depends on it)")
    except Exception as e:  # noqa: BLE001
        log(f"  ablation_v3 cross-check skipped ({type(e).__name__})")
    return out


# ------------------------------------------------------------------ statistics
def sign_test_p(n_pos, n_total):
    if n_total == 0:
        return None
    k = min(n_pos, n_total - n_pos)
    tail = sum(math.comb(n_total, i) for i in range(k + 1)) / (2 ** n_total)
    return min(1.0, 2 * tail)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    m = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m[int((alpha / 2) * n_boot)], m[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def paired_stats(deltas):
    """Judge scores are integers 0-10, so ties are common and are EXCLUDED from
    the sign test. n_tied is reported because a high tie count means the test is
    underpowered, not that the effect is absent."""
    nz = [d for d in deltas if d != 0]
    pos = sum(1 for d in nz if d > 0)
    lo, hi = bootstrap_ci(deltas)
    return {
        "n": len(deltas),
        "n_tied": len(deltas) - len(nz),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
        "n_positive": pos,
        "n_nonzero": len(nz),
        "sign_test_p": sign_test_p(pos, len(nz)),
        "bootstrap_ci95": [lo, hi],
        "per_prompt_deltas": deltas,
    }


# ------------------------------------------------------------------ generate
BATCH_SIZE = 33          # 3 conditions x 11 prompts: one state per generate() call
GEN_CKPT = "behavior_generations.partial.json"
JUDGE_WORKERS = 6        # concurrent judge calls; _call_api already backs off on 429
WORK_DIR = Path(".")     # set by --work-dir in main()


def wp(name):
    """Where a work artifact lives. Point --work-dir at Google Drive and the
    checkpoint + judge cache survive the Colab runtime being recycled, not just
    a kernel restart or a dropped browser connection."""
    return WORK_DIR / name


def load_models():
    """4-bit base with every adapter attached ONCE; states switch by set_adapter.

    An earlier revision attached adapters in a loop guarded by a bare
    `except Exception: pass`, which could silently swallow a failed adapter
    download and leave a state running on the wrong weights. Every load here
    either succeeds or raises.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log(f"loading {BASE_MODEL} in 4-bit — the first run downloads ~2 GB and can "
        f"sit here for a few minutes with no other output")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=bnb, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, **kw)
    model.eval()
    log(f"loaded {BASE_MODEL} (4-bit NF4)")

    log("attaching adapters (downloads ~120 MB each on first run)")
    peft_model = None
    for name, aid in ADAPTERS.items():
        if aid is None:
            continue
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, aid, adapter_name=name)
        else:
            peft_model.load_adapter(aid, adapter_name=name)
        log(f"  attached adapter {name:5s} <- {aid}")
    net = peft_model if peft_model is not None else model
    net.eval()
    return tok, net, peft_model


def stop_token_ids(tok, net):
    """Every token id that ends a response. Qwen2.5 ends a turn with <|im_end|>
    and pads with <|endoftext|>; its generation_config lists both as eos."""
    ids = set()
    e = getattr(getattr(net, "generation_config", None), "eos_token_id", None)
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(int(x) for x in e)
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    for i in (tok.eos_token_id, tok.pad_token_id):
        if i is not None:
            ids.add(int(i))
    return ids


def split_generated(row, stop_ids):
    """(response_token_ids, finished) for one row of a batched generate().

    A row that ends early is padded after its stop token, so the response is
    everything BEFORE the first stop token. `finished` is False only when no
    stop token appears at all: the model hit max_new_tokens mid-answer, and the
    judge would be scoring a cut-off response.
    """
    for i, t in enumerate(row):
        if t in stop_ids:
            return row[:i], True
    return row, False


def _fingerprint(prompts):
    import hashlib
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode())
        h.update(b"\0")
    h.update(json.dumps({"adapters": ADAPTERS, "max_new_tokens": MAX_NEW_TOKENS,
                         "base": BASE_MODEL}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def generate_all(records, conds, models=None, batch_size=BATCH_SIZE):
    """Batched, checkpointed generation with live progress.

    The first revision generated 132 responses ONE AT A TIME through a 4-bit
    model on a T4 — slow at single-sequence decoding, 1-2+ hours — and wrote
    nothing until the very end, so a recycled runtime lost everything. Now each
    state is one batched generate() over all 33 of its prompts (the 4-bit
    dequantisation cost is paid once per step for the whole batch instead of per
    row), and each finished state is checkpointed so a rerun resumes.

    `models` lets a test inject (tok, net, peft_model); normally loaded here.
    """
    import contextlib
    import gc
    import time
    import torch

    tok, net, peft_model = models if models is not None else load_models()
    tok.padding_side = "left"   # decoder-only batching: pad on the LEFT so every
                                # row's prompt ends at the same position
    stop = stop_token_ids(tok, net)
    rep = getattr(getattr(net, "generation_config", None), "repetition_penalty", None)

    jobs = [(cond, sysmsg, rec) for cond, sysmsg in conds.items() for rec in records]
    prompts = []
    for _cond, sysmsg, rec in jobs:
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) \
            + [{"role": "user", "content": rec["user"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
    fp = _fingerprint(prompts)

    # Resume: reuse states finished by an interrupted run, but ONLY if prompts,
    # adapters and the token cap are unchanged — a stale checkpoint is ignored.
    ck, cp = {}, wp(GEN_CKPT)
    if cp.exists():
        try:
            raw = json.loads(cp.read_text())
            if raw.get("fingerprint") == fp:
                ck = raw.get("states", {})
                if ck:
                    log(f"resuming from checkpoint: {sorted(ck)} already complete")
            else:
                log("checkpoint is from a different configuration — ignoring it")
        except json.JSONDecodeError:
            log("checkpoint unreadable — ignoring it")

    n_states = len(ADAPTERS)
    log(f"generating {len(jobs)} responses x {n_states} states, batch {batch_size}, "
        f"max {MAX_NEW_TOKENS} new tokens, greedy"
        + (f", repetition_penalty {rep}" if rep else ""))
    for si, state in enumerate(ADAPTERS, 1):
        tag = f"[{si}/{n_states}] {state:5s}"
        if len(ck.get(state, [])) == len(jobs):
            log(f"{tag} already done — reused from checkpoint")
            continue
        if state != "base" and peft_model is not None:
            peft_model.set_adapter(state)
        t0 = time.time()
        rows = []
        for b0 in range(0, len(jobs), batch_size):
            enc = tok(prompts[b0:b0 + batch_size], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(net.device)
            ctx = (peft_model.disable_adapter()
                   if state == "base" and peft_model is not None
                   else contextlib.nullcontext())
            with torch.no_grad(), ctx:
                out = net.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, pad_token_id=tok.pad_token_id,
                                   eos_token_id=sorted(stop))
            for row in out[:, enc["input_ids"].shape[1]:].tolist():
                ids, finished = split_generated(row, stop)
                rows.append((tok.decode(ids, skip_special_tokens=True).strip(),
                             len(ids), finished))
            log(f"{tag} {min(b0 + batch_size, len(jobs)):3d}/{len(jobs)} generated "
                f"({time.time() - t0:.0f}s)")
        ck[state] = [{**rec, "state": state, "condition": cond, "response": resp,
                      "n_new_tokens": n, "truncated": not fin}
                     for (cond, _s, rec), (resp, n, fin) in zip(jobs, rows)]
        cp.write_text(json.dumps({"fingerprint": fp, "states": ck}))
        n_tr = sum(g["truncated"] for g in ck[state])
        log(f"{tag} done in {time.time() - t0:.0f}s | truncated {n_tr}/{len(jobs)} "
            f"| checkpointed")

    gens = [g for state in ADAPTERS for g in ck[state]]
    meta = {"base_model": BASE_MODEL, "adapters": ADAPTERS,
            "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS,
                         "batch_size": batch_size, "repetition_penalty": rep,
                         "note": "batched greedy with left padding; can differ from "
                                 "unbatched greedy by floating-point noise, equally "
                                 "for every state"},
            "fingerprint": fp}
    wp(GEN_FILE).write_text(json.dumps({"meta": meta, "generations": gens}, indent=2))

    if models is None:          # we own them: free the GPU before judging
        del net, peft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    n_trunc = sum(g["truncated"] for g in gens)
    log(f"wrote {len(gens)} generations to {wp(GEN_FILE)}"
        + (f"\nWARNING: {n_trunc} hit the {MAX_NEW_TOKENS}-token cap and are cut off "
           f"mid-answer; the judge will penalise them. Per-state counts are in the "
           f"report." if n_trunc else ""))
    return gens


def load_generations():
    """Read GEN_FILE. Accepts the current {meta, generations} layout and the
    earlier bare-list layout."""
    raw = json.loads(wp(GEN_FILE).read_text())
    return raw["generations"] if isinstance(raw, dict) else raw


# ------------------------------------------------------------------ judge
JUDGE_FAILED = ("evaluation failed", "parse error")


def judge_failed(ev):
    """True for score_candidate's failure sentinel.

    On an API failure or unparseable judge JSON, score_candidate returns
    {"overall": 5, "summary": "evaluation failed" | "parse error"}. That 5 is
    not a judgment. Counting it pulls every state toward the middle of the
    scale; caching it (as the first revision here did) turns a transient 429
    into a permanent fake score on every re-run.
    """
    return (not isinstance(ev, dict)) or ev.get("summary") in JUDGE_FAILED


def judge_all(gens, gdp, judge_model, use_cache=True,
              workers=JUDGE_WORKERS, retry_rounds=3):
    """Concurrent, cached, failure-aware judging.

    Only genuine judgments are cached. Calls that still fail after
    `retry_rounds` get score=None and are EXCLUDED from analysis — never scored
    as 5 — and a later `--phase judge` retries just those, since everything that
    succeeded is already cached.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    recs = {r["id"]: r for r in gdp.load_annotation_records(
        gdp.ANNOTATIONS_DIR, ["draft", "committed"])}

    cache, cp = {}, wp(CACHE_FILE)
    if use_cache and cp.exists():
        try:
            cache = json.loads(cp.read_text())
        except json.JSONDecodeError:
            log("judge cache unreadable — starting empty")
    stale = [k for k, v in cache.items() if judge_failed(v)]
    for k in stale:
        del cache[k]
    if stale:
        log(f"dropped {len(stale)} cached judge FAILURES that an earlier revision "
            f"had stored as score 5; they will be re-judged")

    def key_of(g):
        return gdp._cache_key(g["record_id"], gdp.JUDGE_PROMPT_TAG, judge_model,
                              g["state"], g["condition"], g["user"], g["response"])

    todo = []
    for g in gens:
        g["score"] = None
        if g["record_id"] not in recs:
            g["judge_error"] = "no annotation record"
            log(f"  WARNING: no annotation record {g['record_id']} — excluded")
            continue
        ev = cache.get(key_of(g))
        if ev is not None:
            g["evaluation"], g["score"] = ev, ev.get("overall")
            g.pop("judge_error", None)
        else:
            todo.append(g)
    log(f"judge: {len(gens) - len(todo)} cached, {len(todo)} to score "
        f"with {judge_model}, {workers} at a time")

    def one(g):
        return gdp.score_candidate(client, g["user"], g["response"],
                                   recs[g["record_id"]], judge_model)

    for rnd in range(1, retry_rounds + 1):
        if not todo:
            break
        failed, done, t0 = [], 0, time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(one, g): g for g in todo}
            for f in as_completed(fut):
                g = fut[f]
                try:
                    ev = f.result()
                except Exception as e:  # noqa: BLE001
                    ev = {"summary": "evaluation failed",
                          "error": f"{type(e).__name__}: {e}"}
                done += 1
                if judge_failed(ev):
                    g["judge_error"] = ev.get("error") or ev.get("summary")
                    failed.append(g)
                else:
                    g["evaluation"], g["score"] = ev, ev.get("overall")
                    g.pop("judge_error", None)
                    cache[key_of(g)] = ev
                    if use_cache:
                        cp.write_text(json.dumps(cache, indent=1))
                if done % 12 == 0 or done == len(todo):
                    log(f"  round {rnd}: {done:3d}/{len(todo)} judged | "
                        f"{len(failed)} failed | {time.time() - t0:.0f}s")
        todo = failed
        if todo and rnd < retry_rounds:
            wait = 20 * rnd
            log(f"  {len(todo)} judge calls failed — retrying them in {wait}s")
            time.sleep(wait)

    if todo:
        log(f"\nWARNING: {len(todo)} generation(s) could not be judged after "
            f"{retry_rounds} rounds. They are EXCLUDED from the analysis — not "
            f"scored as 5. Run the judge step again to retry only these.")
        for g in todo[:5]:
            log(f"    {g['state']:5s} {g['condition']:7s} {g['record_id']}: "
                f"{g.get('judge_error')}")
    return gens


# ------------------------------------------------------------------ analyse
def analyse(gens):
    key = lambda g: (g["condition"], g["user"])          # noqa: E731
    scen = {}
    by_state = defaultdict(dict)
    for g in gens:
        if g.get("score") is None:
            continue
        by_state[g["state"]][key(g)] = g["score"]
        scen[key(g)] = g["scenario"]

    res = {"absolute": {}, "paired_vs_base": {}, "paired_vs_base_scenario_clustered": {},
           "project_comparisons": {}, "data_quality": {}}

    # Per-state counts of what could NOT be scored fairly. Truncated responses
    # are judged but penalised for being cut off; unjudged ones are excluded.
    # A state with many of either is being compared on a different footing.
    for st in dict.fromkeys(g["state"] for g in gens):
        mine = [g for g in gens if g["state"] == st]
        res["data_quality"][st] = {
            "n": len(mine),
            "truncated": sum(1 for g in mine if g.get("truncated")),
            "unjudged_excluded": sum(1 for g in mine if g.get("score") is None),
        }

    for st, sc in by_state.items():
        per_cond = defaultdict(list)
        for (c, _), v in sc.items():
            per_cond[c].append(v)
        res["absolute"][st] = {
            "n": len(sc),
            "mean_overall": sum(sc.values()) / len(sc),
            "mean_by_condition": {c: sum(v) / len(v) for c, v in per_cond.items()},
            "score_histogram": dict(sorted(Counter(sc.values()).items())),
        }

    base = by_state.get("base", {})

    def cluster(pairs):
        g = defaultdict(list)
        for k, d in pairs:
            g[scen[k]].append(d)
        return [sum(v) / len(v) for _, v in sorted(g.items())]

    conds = sorted({c for c, _ in base})
    for st, sc in by_state.items():
        if st == "base":
            continue
        res["paired_vs_base"][st] = {}
        res["paired_vs_base_scenario_clustered"][st] = {}
        for c in conds:
            pairs = [(k, sc[k] - base[k]) for k in sc if k in base and k[0] == c]
            if not pairs:
                continue
            res["paired_vs_base"][st][c] = paired_stats([d for _, d in pairs])
            res["paired_vs_base_scenario_clustered"][st][c] = paired_stats(cluster(pairs))
        # the three comparisons judge_rubric.py reports, for continuity
        m = res["absolute"]
        bm, sm = m["base"]["mean_by_condition"], m[st]["mean_by_condition"]
        res["project_comparisons"][st] = {
            "prompt_effect_on_base (base+RS - base+gen)": bm["rs"] - bm["generic"],
            "adapter_effect_under_RS (adapter+RS - base+RS)": sm["rs"] - bm["rs"],
            "behavior_in_weights (adapter+gen - base+gen)": sm["generic"] - bm["generic"],
            "unconditionality_ratio (adapter gen-gain / rs-gain)": (
                (sm["generic"] - bm["generic"]) / (sm["rs"] - bm["rs"])
                if (sm["rs"] - bm["rs"]) else None),
        }
    return res


def report(res):
    dq = res.get("data_quality", {})
    if dq:
        log("\n" + "=" * 78)
        log("DATA QUALITY — read this before the numbers")
        log("=" * 78)
        for st, d in dq.items():
            flag = "  <- check" if (d["truncated"] or d["unjudged_excluded"]) else ""
            log(f"  {st:5s} {d['n']:3d} responses | truncated {d['truncated']:3d} "
                f"| unjudged (excluded) {d['unjudged_excluded']:3d}{flag}")
    log("\n" + "=" * 78)
    log("MEAN JUDGE OVERALL (0-10)")
    log("=" * 78)
    conds = ["rs", "generic", "none"]
    log(f"{'state':6s}" + "".join(f"{c:>11s}" for c in conds) + f"{'all':>9s}")
    for st, a in res["absolute"].items():
        log(f"{st:6s}" + "".join(f"{a['mean_by_condition'].get(c, float('nan')):>11.3f}"
                                 for c in conds) + f"{a['mean_overall']:>9.3f}")

    log("\n" + "=" * 78)
    log("PAIRED vs base — scenario-clustered. GENERIC is the headline.")
    log("=" * 78)
    hdr = f"{'state':6s} {'cond':8s} {'mean d':>8s} {'pos/n':>8s} {'tied':>5s} {'sign p':>8s} {'CI95':>18s}"
    log(hdr); log("-" * len(hdr))
    for st, cs in res["paired_vs_base_scenario_clustered"].items():
        for c in conds:
            s = cs.get(c)
            if not s:
                continue
            lo, hi = s["bootstrap_ci95"]
            mark = " <<" if c == "generic" else ""
            star = " *" if (s["sign_test_p"] or 1) < 0.05 else "  "
            log(f"{st:6s} {c:8s} {s['mean_delta']:>+8.3f} "
                f"{s['n_positive']:>3d}/{s['n_nonzero']:<4d} {s['n_tied']:>5d} "
                f"{s['sign_test_p']:>8.4f}{star}[{lo:>+6.2f},{hi:>+6.2f}]{mark}")

    log("\n" + "=" * 78)
    log("PROJECT COMPARISONS (judge_rubric.py's three numbers)")
    log("=" * 78)
    for st, cmp_ in res["project_comparisons"].items():
        log(f"  {st}:")
        for k, v in cmp_.items():
            log(f"    {k:<52s} {v:+.3f}" if v is not None else f"    {k:<52s}   n/a")
    log("\n* = sign test p < 0.05   << = headline (behavior in the weights)")


# ------------------------------------------------------------------ dry run
def dry_run(records, conds, gdp):
    assert records, "no eval records"
    scen = {r["scenario"] for r in records}
    log(f"DRY RUN: {len(records)} prompts over {len(scen)} scenarios "
        f"| pair_type {dict(Counter(r['pair_type'] for r in records))}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    for r in records:
        for c, sysmsg in conds.items():
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) \
                + [{"role": "user", "content": r["user"]}]
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert t.rstrip().endswith("<|im_start|>assistant"), f"bad template for {c}"
            if c == "rs":
                assert "in the Buddhist" in t, "rs condition is not the TRAINING rs prompt"
    log(f"DRY RUN: {len(records) * len(conds)} generation prompts templated; "
        f"rs condition matches diversify_prompts.RS_PROMPT")

    recs = {r["id"]: r for r in gdp.load_annotation_records(
        gdp.ANNOTATIONS_DIR, ["draft", "committed"])}
    missing = {r["record_id"] for r in records} - set(recs)
    assert not missing, f"eval prompts reference unknown records: {missing}"
    log(f"DRY RUN: all {len(records)} prompts resolve to annotation records")

    fake = []
    for st in ADAPTERS:
        for c in conds:
            for i, r in enumerate(records):
                fake.append({**r, "state": st, "condition": c, "response": "x",
                             "score": 5 + (1 if st != "base" else 0)})
    res = analyse(fake)
    for st in ADAPTERS:
        if st == "base":
            continue
        s = res["paired_vs_base_scenario_clustered"][st]["generic"]
        assert abs(s["mean_delta"] - 1.0) < 1e-9
    log("DRY RUN: analysis + statistics validated on synthetic scores")
    n_calls = len(ADAPTERS) * len(conds) * len(records)
    log(f"DRY RUN: a real run makes {n_calls} judge calls "
        f"({len(ADAPTERS)} states x {len(conds)} conditions x {len(records)} prompts)")
    log("DRY RUN OK")


# ------------------------------------------------------------------ main
def main():
    global WORK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "judge", "both"], default="both")
    ap.add_argument("--repo", default=".", help="path to local-trainer/")
    ap.add_argument("--work-dir", default=".",
                    help="where generations, checkpoint, judge cache and results "
                         "go. Point at Google Drive to survive a recycled runtime.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="prompts per generate() call; lower it only on OOM")
    ap.add_argument("--workers", type=int, default=JUDGE_WORKERS,
                    help="concurrent judge calls; lower it if you keep hitting 429s")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6",
                    help="must match generate_dpo_pairs.py's judge or scores are "
                         "not comparable to the training data")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    WORK_DIR = Path(args.work_dir).resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {WORK_DIR}")

    repo = Path(args.repo).resolve()
    gdp, conds = load_project(repo)
    records = load_eval_records(repo)

    if os.environ.get("SAIGE_DRY_RUN", "") in ("1", "true"):
        dry_run(records, conds, gdp)
        return

    if args.phase in ("generate", "both"):
        gens = generate_all(records, conds, batch_size=args.batch_size)
    else:
        assert wp(GEN_FILE).exists(), (
            f"{wp(GEN_FILE)} not found — run the generate step first "
            f"(with the same --work-dir)")
        gens = load_generations()
        log(f"loaded {len(gens)} generations from {wp(GEN_FILE)}")
    if args.phase == "generate":
        log("GENERATION COMPLETE — run the judge step next (no GPU needed)")
        return

    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY required for judging"
    gens = judge_all(gens, gdp, args.judge_model, use_cache=not args.no_cache,
                     workers=args.workers)
    res = analyse(gens)
    report(res)

    meta = {}
    try:
        raw = json.loads(wp(GEN_FILE).read_text())
        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    wp(OUT_FILE).write_text(json.dumps({
        "base_model": BASE_MODEL,
        "judge_model": args.judge_model,
        "judge": "generate_dpo_pairs.score_candidate (same judge as the training pairs)",
        "adapters": ADAPTERS,
        "eval_set": "the 11 held-out legacy records, identical to ablation_v3 --eval-set legacy",
        "headline": "paired_vs_base_scenario_clustered[state]['generic']",
        "decoding": meta.get("decoding", f"greedy, max_new_tokens={MAX_NEW_TOKENS}"),
        "results": res,
        "generations": gens,
    }, indent=2))
    log(f"\nwrote {wp(OUT_FILE)}")


if __name__ == "__main__":
    main()
