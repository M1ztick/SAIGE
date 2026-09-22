# SAIGE 2×2 ablation results

Logprob-margin evaluations of the SAIGE DPO adapters: for each state (base model +
adapters) × prompt condition (rs / generic / none) × eval record,

    margin = logp(chosen | prompt) − logp(rejected | prompt)

> ## ⚠️ Correction notice (supersedes everything below the fold)
>
> Two defects were found in the v1/v2 ablations. **The published margin tables are
> withdrawn**, and the conclusions drawn from them — in particular "run5 memorized
> but did not generalize" — **are not supported by the evidence that produced
> them.** They are neither confirmed nor refuted; the experiment that could decide
> the question has not been run yet.
>
> ### Defect 1 — the scored completion was malformed
>
> `ablation_v2.py`'s `score_pair` built the completion with
> `apply_chat_template(completion_msgs, add_generation_prompt=False)`. On Qwen2.5,
> rendering an assistant-only message list re-emits the template's **default system
> turn** and a **second assistant header**:
>
> ```
> '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful
>  assistant.<|im_end|>\n<|im_start|>assistant\nThat sounds draining.<|im_end|>\n'
> ```
>
> Appended to a prompt already ending in `<|im_start|>assistant\n`, this scored
> every response *after a malformed mid-conversation second system turn*, and
> padded each completion with a constant **24 junk tokens**. Both the total-nat and
> the per-token margins in `ablation_results_v2.json` are wrong in absolute terms.
>
> This is why `ablation_results_v2.json` and `ablation_results_run5.json` disagree
> by up to 16 nats on individual records while agreeing on deltas: the junk block is
> identical across states, so it largely cancels in a difference. The claim in the
> previous README that the two "agree within noise" was true only of the deltas.
> **The absolute claim that "the base model's prior against the chosen style is
> 12–16 nats deep" is withdrawn.**
>
> ### Defect 2 — the eval set could not observe what run5 learned
>
> run5 was trained on 16 pairs gated at judge `score_delta >= 3`. It was evaluated
> on the 11 grouped-split records of the **old** dataset, whose pairs sit at **mean
> judge delta 0.42** under that same 72B judge — "good vs. slightly better". The
> eval and training distributions did not match, so a flat result is what you would
> expect *whether or not* the adapter learned something real. A data-quantity
> explanation and an eval-mismatch explanation are not distinguishable from these
> numbers, so "quantity is the remaining constraint" is **untested**, not shown.
>
> ### Defect 3 — the statistic was noise-dominated
>
> v2 reported the mean of per-record total-nat margins. Those margins span −150 to
> +130 and are driven by response length and style; preference accuracy sat at
> 0.36–0.45 for *every* state because the sign is set by the base model's prior, not
> by the adapter. v3 reports **paired per-record deltas vs base**, which cancel the
> per-record offset, with an exact sign test and a bootstrap CI.
>
> ### Defect 4 — a prompt confound (fixed going forward)
>
> Through v4, the prompt that generated the **chosen** responses
> (`SAIGE_SYSTEM_PROMPT`) and the `rs` **evaluation condition** (`RS_PROMPT`)
> differed by a trailing newline — one extra token inside the system turn. The `rs`
> condition was therefore never byte-identical to the prompt the chosen text was
> produced under. Aligned in `ablation_v3.py`, `eval_generation.py` and
> `generate_rejects_v5.py`.
>
> ### What is still standing
>
> One result survives the recomputation of deltas, because deltas are largely
> insulated from Defect 1: **run3 moved every one of the 11 records in the positive
> direction under both the rs and generic conditions** (+3 to +9 nats).
>
> The significance of that sweep was previously quoted as p ≈ 0.001, which is
> **too generous**. A sign test assumes independent records, and these 11 span
> only **9 distinct (record_id, prompt_type) scenarios**: saige-rs-007 and
> saige-rs-009 each contribute two records generated from the same annotated
> situation, so those are near-paraphrases, not independent trials. Clustered on
> scenario the same sweep is **p ≈ 0.004**. Still a real, consistent effect —
> just not the number previously published. `ablation_v3.py` now reports both and
> marks the clustered one as the headline.
>
> Even at p ≈ 0.004, the effect's uniformity across records *and* prompt
> conditions is equally consistent with a generic cause (opener style, length) as
> with Right Speech, and `ablation_v3.py` cannot tell those apart.
> `eval_generation.py` is the instrument that can.
>
> ### What replaces this
>
> | Script | Purpose |
> |---|---|
> | `ablation_v3.py` | **Canonical scorer.** Correct templating, paired deltas + sign test + bootstrap CI, two eval sets, per-stratum breakdown. `ABLATION_DRY_RUN=1` validates on CPU. |
> | `eval_generation.py` | **Behavioral eval.** Generates from each state on held-out prompts and scores with the same 72B judge and verbatim rubric that gated the data. Logprob margins on fixed text are not behavior. |
> | `generate_rejects_v5.py` | Generation pipeline with a scenario- and persona-grouped held-out split (in the dataset repo). |
> | `train_dpo_v6.py` | Trainer consuming the matched train/eval split (in the dataset repo). |
>
> No corrected numbers are published yet — the jobs that produce them have not been
> run. **Do not cite the tables below.**

---

## Files

| File | Status |
|---|---|
| `ablation_v3.py` | **Canonical.** Use this. |
| `eval_generation.py` | **Canonical** behavioral eval. Use this. |
| `ablation_results_v3.json` | Not yet generated. |
| `generation_eval_results.json` | Not yet generated. |
| `ablation_v2.py` | ⚠️ Withdrawn — malformed completion templating (Defect 1). |
| `ablation_results_v2.json` | ⚠️ Withdrawn — absolute margins invalid. |
| `ablation_run5.py` | ⚠️ Withdrawn — superseded. |
| `ablation_results_run5.json` | ⚠️ Withdrawn. Its absolute numbers differ from v2's by up to 16 nats/record for the reason in Defect 1. |
| `ablation_results.json` | ⚠️ Withdrawn — v1, earlier scoring implementation. |

## Method (v3)

- Eval records: `--eval-set legacy` (the 11 old-dataset records, kept for
  continuity only), `--eval-set heldout` (high-contrast delta ≥ 3 pairs from the
  v5 split — **use this for conclusions**), or `both`.
- **What the legacy set is.** It IS annotation-derived: every row carries a
  `record_id` into `saige-rs-001..012` and a `prompt_type` taken verbatim from
  that record's `example_prompt_types`. What is weak is the rejected side, not
  the provenance. Of the 11 scored records, **10 are `pair_type: ranked`** (two
  plausible answers ranked against each other) and 1 is `misreading` (Right
  Speech vs an engineered failure); mean `score_delta` is 1.55 under the original
  scoring, and the set covers 7 of the 12 annotation records. It remains a valid
  matched eval for an adapter that was *trained* on this dataset.
- Headline statistic: the paired delta vs base, same record and same condition,
  with an exact two-sided sign test and a percentile bootstrap CI. Absolute
  margins and per-token margins are also recorded.
- **Scenario clustering.** A scenario is a `(record_id, prompt_type)` cell of the
  annotation grid. Where several eval records come from one cell they are
  regenerations of the same annotated situation and are not independent, so the
  script also reports the sign test with deltas averaged within a scenario — one
  vote per scenario — and names that the headline whenever duplicates exist.
  The legacy set has 11 records over 9 scenarios, so this applies to it.
- The held-out set carries two strata, reported separately:
  `unseen_scenario` and `unseen_scenario_persona` (a misreading persona the
  adapter was never trained against — the harder test).

### The "none" condition is not "no system prompt"

Qwen2.5's chat template injects its own default system turn when no system message
is supplied. The `none` condition is therefore **"Qwen's default system prompt"**.
The key is kept for table continuity; the true meaning is recorded in
`condition_notes` in the output JSON.

### Reading the rs/generic gap

A flat rs-minus-generic gap means an adapter shifted all prompt conditions in
parallel. For an adapter that shifted nothing, a flat gap is evidence of nothing.
The earlier reading — that a flat gap across every adapter demonstrated a real
"prompt-independent experiential mechanism" — does not follow, and is withdrawn.
