# SAIGE — Development Guidelines

## Code Quality Standards

### Module Structure and Size
- Python modules are kept under ~1000 lines; extraction is preferred over growth (scorers.py was explicitly split from saige_to_sft_v2.py)
- TypeScript files follow a single-responsibility pattern: one module per concern (harm_detection.ts, buddhist_principles.ts, worker.ts are separate)
- Long files use horizontal section dividers with consistent width

Python divider style (used in all active Python files):
```python
# ─────────────────────────────────────────────────────────────
# Section Title
# ─────────────────────────────────────────────────────────────
```

TypeScript divider style (used in both worker-side files):
```typescript
// ── Section Title ─────────────────────────────────────────────────────────────
```

### File Headers
- All Python scripts begin with a shebang + module docstring that describes purpose, pipeline flow, and usage examples
- TypeScript files begin with a single-line filename comment, followed by a brief purpose comment and a CHANGELOG block

Python header pattern:
```python
#!/usr/bin/env python3
"""
module_name.py — Short description

Pipeline:
  1. Step one
  2. Step two

Usage:
  python module_name.py --flag value
"""
```

TypeScript header pattern:
```typescript
// filename.ts
// Brief description
//
// CHANGELOG:
//   + feature_name() — description
```

---

## Naming Conventions

### Python
- Functions: `snake_case` verbs — `score_candidate`, `build_ranked_pair`, `load_annotation_records`
- Classes: `PascalCase` nouns — `TextCleaner`, `CalibrationScorer`, `QualityMetrics`, `CleaningReport`
- Constants: `UPPER_SNAKE_CASE` — `ANNOTATIONS_DIR`, `CACHE_FILE`, `MIN_SCORE_DELTA`, `JUDGE_PROMPT_TAG`
- Dataclasses used for structured result containers (never plain dicts for return types)

### TypeScript
- Functions: `camelCase` verbs — `detectDeception`, `scoreAhimsa`, `assessBuddhistPrinciples`
- Interfaces: `PascalCase` — `BuddhistPrincipleScores`, `BuddhistAssessment`, `HarmAssessment`
- Private helpers: prefixed with `score`, `detect`, `contains`, `is`, `extract`, `generate`
- Regex constants: `UPPER_SNAKE_CASE_PATTERN` — `MATH_CONTRADICTION_PATTERN`, `SKY_COLOR_CONTRADICTION_PATTERN`

---

## Structural Conventions

### Scoring Functions (TypeScript — buddhist_principles.ts, harm_detection.ts)
All principle/dimension scorers follow an identical structure:
1. Initialize `let score = 5.0` (neutral midpoint)
2. Apply positive indicator arrays: `{ pattern: /regex/, points: float }`
3. Apply negative indicator arrays: `{ pattern: /regex/, points: negative_float }`
4. Apply scenario-context adjustments (vulnerability, mood checks)
5. Return `Math.max(0, Math.min(10, score))`

Example:
```typescript
function scoreAhimsa(response: string, scenario?: Scenario): number {
  let score = 5.0;
  const nonHarmIndicators = [
    { pattern: /\b(safe|safety|protect)\b/gi, points: 1.0 },
  ];
  for (const { pattern, points } of nonHarmIndicators) {
    const matches = response.match(pattern);
    if (matches) score += Math.min(matches.length * points, 2.0);
  }
  return Math.max(0, Math.min(10, score));
}
```

### Scoring Classes (Python — scorers.py)
All scorer classes use `@classmethod` methods only (no instance state):
```python
class CalibrationScorer:
    CALIBRATION_TARGETS = { ... }  # class-level config

    @classmethod
    def classify_prompt(cls, context: str, difficulty: int, person_state: dict) -> str:
        ...

    @classmethod
    def score(cls, response: str, ...) -> float:
        ...
```

### Dataclasses for Result Types
Structured results always use `@dataclass`, never plain dicts:
```python
@dataclass
class CleaningReport:
    experience_id: int = 0
    typos_fixed: int = 0
    kept: bool = True
    rejection_reason: str = ""
```

### API Helper Pattern (generate_dpo_pairs.py)
External API calls are always wrapped with:
- Retry loop with exponential backoff
- Specific exception handling (`RateLimitError` → wait, `APIError` → retry with backoff)
- `Optional[str]` return type (None on failure, never raises to caller)

```python
def _call_api(..., retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            ...
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt * 5)
        except anthropic.APIError as e:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None
```

### Cache Pattern
Expensive API calls are always cached to a JSON file:
- Cache key is a short SHA-256 hash: `hashlib.sha256(raw.encode()).hexdigest()[:20]`
- Cache keys incorporate content hashes of prompts/rubrics so edits self-invalidate stale entries
- Cache is loaded at startup, saved in a `finally` block at teardown
- `--no-cache` flag always provided to bypass

```python
CACHE_FILE = Path(__file__).parent / ".dpo_cache.json"

cache_key = _cache_key(record_id, *parts)
if cache_key in cache:
    result = cache[cache_key]
else:
    result = expensive_api_call(...)
    cache[cache_key] = result
```

### JSON Output Pattern
All JSONL output uses `ensure_ascii=False`:
```python
out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
```

JSON config files are written with `indent=2, ensure_ascii=False`.

---

## Semantic Patterns

### Score Clamping — Universal
Both Python and TypeScript apply the same `[0, 10]` clamp at every scoring boundary:
- TypeScript: `Math.max(0, Math.min(10, score))`
- Python: `max(0, min(10, score))`
Never use raw scores without clamping.

### Score Cap Propagation (Anti-inflation Pattern)
The judge score reconciliation in generate_dpo_pairs.py deliberately pulls scores down when sub-scores are weak:
```python
def _reconcile_overall(judge_overall: int, sub_scores: list) -> int:
    worst = min(sub_scores)
    mean  = sum(sub_scores) / len(sub_scores)
    blended = round(0.6 * worst + 0.4 * mean)
    return max(0, min(10, min(judge_overall, blended)))
```
The worst dimension is always weighted 60% — the system is explicitly designed to prevent score inflation.

### Minimum Threshold Gating
Pairs are only produced when score delta meets a minimum:
```python
MIN_SCORE_DELTA = 1
if delta < MIN_SCORE_DELTA:
    return None
```
This pattern repeats in both `build_ranked_pair` and `_process_misreading`.

### Scenario-Context Scoring Modifier
All scoring functions check `scenario.person_state.vulnerability` and `scenario.person_state.mood` to modulate scores. Vulnerability levels: `low`, `high`, `extreme`. Mood states: `neutral`, `depressed`, `anxious`, `distressed`, `desperate`, `testing`.

### Presence-Before-Advice Check
A recurring ethical pattern enforced in `scorePresence()` and `generateLesson()`: in emotional contexts (high/extreme vulnerability or distressed moods), acknowledgment MUST precede advice. Advice-pivot in an emotional opening context incurs a 3.0 penalty to karuna. This is the primary "timely speech" operationalization.

### Regex Pattern Organization (TypeScript)
Named regex patterns extracted as module-level constants when reused across multiple functions:
```typescript
const MATH_CONTRADICTION_PATTERN = /2\s*\+\s*2\s*=\s*5/i;
```
Inline regex used when single-use.

### SAIGE vs Baseline Generation
When generating DPO candidates, SAIGE system prompt candidates are always generated first, then a smaller number (half) of baseline candidates. They are concatenated — training learns from the contrast between SAIGE-prompted and generic-prompted outputs.

---

## CLI Conventions

### Python CLI (argparse)
All scripts use `argparse.ArgumentParser` with:
- `formatter_class=argparse.RawDescriptionHelpFormatter`
- `epilog` containing concrete usage examples
- `--dry-run` flag for preview without API/disk side effects
- `--no-cache` flag for forced regeneration
- Sensible defaults for all optional parameters (documented in help strings)

### Progress Output Style
```
SAIGE DPO Pair Generator
============================================================

Loading annotation records from /path...
  Loaded: saige-rs-001 — Title

──────────────────────────────────────────────────────────────
Record: saige-rs-001 — Title
──────────────────────────────────────────────────────────────
  Prompt type: "..."
    [1/2] "preview of message..."
      Generated 5 candidates (3 SAIGE + 2 baseline)
      Ranked pair: chosen=7 rejected=4 delta=3
```
- Warnings go to `sys.stderr`
- Progress/informational output goes to `sys.stdout`
- Wide separator lines use `─` (U+2500) or `═` (U+2550) for visual hierarchy

### Legacy Script Style (convert_rl_to_sft.py)
Older scripts use emoji-prefixed print statements (📖, ✍️, ✅, 📋). New scripts do not — this was superseded by the plainer structured output style.

---

## Worker API Patterns (TypeScript)

### Endpoint Guard Pattern
Each route checks `url.pathname` and, for mutations, `request.method`:
```typescript
if (url.pathname === '/api/simulate-outcome' && request.method === 'POST') {
```

### Safe JSON Parsing
Always use `safeJsonParse` for D1 JSON columns — never raw `JSON.parse`:
```typescript
function safeJsonParse<T>(jsonString: string | null | undefined, defaultValue: T): T {
  if (!jsonString) return defaultValue;
  try { return JSON.parse(jsonString); }
  catch { return defaultValue; }
}
```

### CORS Headers
CORS headers are applied to every response (including error responses):
```typescript
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};
```
Preflight OPTIONS requests are handled before routing.

### Error Responses
All errors use `Response.json` with appropriate status codes. Error messages expose `error.message` from caught `Error` instances, with `'Unknown error'` as fallback:
```typescript
return Response.json({
  error: error instanceof Error ? error.message : 'Unknown error'
}, { status: 500, headers: corsHeaders });
```

---

## Data Format Conventions

### DPO Pair Schema
Every DPO pair in `dpo_pairs.jsonl` contains:
- `prompt`: messages array in `[{role, content}]` format — system + user only
- `chosen`/`rejected`: single-element arrays `[{role: "assistant", content: "..."}]`
- Metadata: `record_id`, `record_title`, `path_factor`, `canonical_id`, `prompt_type`, `pair_type`
- Scores: `chosen_score`, `rejected_score`, `score_delta`
- Full evaluations: `chosen_evaluation`, `rejected_evaluation` (objects with `flaws`, `scores`, `overall`, `summary`)

### Annotation File Schema
Each `annotations/saige-rs-*.json` file must contain:
- `id`, `title`, `annotation_status` (pending/draft/committed/reviewed)
- `path_factor` — which Noble Eightfold Path factor this covers
- `canonical_id` — Buddhist text reference (e.g., `SN 45.8`)
- `core_principle` — the specific principle being tested
- `behavior_targets` — list of specific behaviors the pair should elicit/avoid
- `example_prompt_types` — list of prompt type descriptions for expansion
- `unsafe_misreadings` — list of misreading names (keys into `_MISREADING_PERSONAS`)
- `evaluation_questions` — per-dimension scoring questions for the LLM judge

### Annotation Status Lifecycle
`pending` → `draft` → `committed` → `reviewed`
Only `draft` and `committed` records are processed by default.
