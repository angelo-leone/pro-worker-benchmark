# Human Validation Pilot — Single-Annotator Protocol

**Hand-off doc.** Self-contained brief for a helper Claude to assist the paper author in running a one-person human-validation pilot for the Pro-Worker AI Benchmark. The author does the scoring; the helper Claude generates the sampling, the scoring file, and the analysis. Total author time: ~90 minutes.

---

## Why this exists

The Pro-Worker AI Benchmark uses a three-judge LLM panel with median aggregation. Per-instance inter-rater reliability (IRR) on two dimensions (Uncertainty Transparency, Appropriate Reliance) is below 20% exact agreement, which a NeurIPS reviewer is likely to challenge. The paper's current defense is structural (population-level averaging, median aggregation, future human validation as Section 7 priority). A small expert-annotator pilot would convert that structural defense into an empirical anchor — a number to cite — and substantially strengthen the IRR mitigation in §6.3.

The pilot is intentionally small (N=30, single annotator) so it can be done before the May 6 NeurIPS deadline. It is positioned in the paper as a *pilot*, not a full validation study; the full study remains future work.

---

## Goal

Produce these numbers, defensible and reproducible, ready to slot into the paper:

1. **Exact agreement** between author score and LLM-judge median (overall, and on UT + AR\_d specifically).
2. **Adjacent agreement** (within ±1 on the 0-3 scale) — the standard metric for ordinal coding, and the most informative for a 4-point scale.
3. **Cohen's quadratic weighted kappa** between author and LLM-judge median.
4. **Mean absolute deviation** between author and LLM-judge median.

Reported in §6.3 (IRR mitigation) and Appendix C.1 (full mitigation argument). Sample, scores, and analysis released in supplementary materials.

---

## Repo context

- Project root: `/Users/angelo.leone/Documents/pro-worker-benchmark`
- Per-run benchmark results: `dataset_staging/results/openai_*.json` (one file per model × variant; each file has top-level `layer1`, `layer2`, `layer3` keys)
- Layer 1 structure: `data['layer1'][dimension]` is a list of dicts with keys `prompt_id`, `prompt`, `domain`, `difficulty`, `primary_dimension`, `n_runs`, `runs`, `mean_score`. Each entry in `runs` has `response` (the model's text) and a per-judge scoring breakdown.
- Rubrics: `rubrics/dimension_rubrics.yaml` (10 L1 dimensions plus adversarial)
- Few-shot calibration examples: `rubrics/examples/*.yaml`
- 7 evaluated models: DeepSeek V3.2, Devstral-2 123B, Gemma 4 31B, GLM 5.1, GPT-oss 120B, Nemotron-Cascade 30B, Qwen3.5 27B
- 2 conditions per model: `baseline`, `with_system_prompt`

The latest file per (model, variant) containing all three layers is what the analysis pipeline uses; same selection rule applies here.

---

## Protocol overview

| Phase | Who | Time | Output |
|-------|-----|------|--------|
| 1. Sample | Helper Claude | 10 min | `pilot/pilot_items.md`, `pilot/pilot_ground_truth.json` |
| 2. Score blind | Author | 60-75 min | `pilot/pilot_scores.csv` |
| 3. Analyze | Helper Claude | 10 min | `pilot/pilot_results.md` |
| 4. Integrate | Helper Claude | 10 min | Paper §6.3 + App C.1 patches |

Author does only Phase 2. Phases 1, 3, 4 are scripted.

---

## Phase 1 — Sampling

Stratification (target N=30):

| Dimension | N | Reason |
|-----------|---|--------|
| Uncertainty Transparency (UT) | 6 | Lowest IRR (15.0%); validation matters most here |
| Appropriate Reliance (AR\_d) | 6 | Second-lowest IRR (18.7%) |
| Metacognitive Calibration (MC) | 4 | Third-lowest (42.3%) |
| Cognitive Forcing (CF) | 3 | High-IRR control + load-bearing dimension |
| Anti-Sycophancy (AS) | 3 | Highest-IRR control |
| Complementarity (CO) | 2 | Load-bearing |
| Ethical Surfacing (ES) | 2 | High-IRR |
| Contrastive Explanation (CE) | 2 | Mid-IRR |
| Skill Preservation (SP) | 1 | Coverage |
| Draft Annotation (DA) | 1 | Coverage |
| **Total** | **30** | |

Within each dimension, sample with `random.seed(42)`:

- 50% baseline / 50% prompted, balanced
- Within condition, prefer one prompt per model where possible (to avoid model-specific bias)
- Within prompt, pick run index 0 (first run); use run 1 only if run 0 is missing

For each sampled item, extract:

- `item_id` (1..30)
- `dimension` (e.g., `uncertainty_transparency`)
- `model` (e.g., `GLM 5.1`)
- `condition` (`baseline` / `with_system_prompt`)
- `prompt_id` (e.g., `ut_07`)
- `user_prompt` (full text, from the prompt YAML)
- `model_response` (full text, from `runs[0].response`)
- **(hidden from author during scoring)** `llm_judge_median` (the median of the three judge scores for this run)

Write two files:

1. `pilot/pilot_items.md` — author-facing, one section per item, no LLM scores shown. Each section has:
   ```
   ## Item N (Dimension: <name>)

   **User prompt:**
   > <full prompt>

   **Model response:**
   > <full response>

   **Rubric for <dimension>:**
   - 3 (Strong): <anchor>
   - 2 (Partial): <anchor>
   - 1 (Weak): <anchor>
   - 0 (Fail): <anchor>

   **Your score (0-3):** ___
   **Your note (optional, ~10 words):** ___
   ```
   (Pull rubric anchors verbatim from `rubrics/dimension_rubrics.yaml`.)

2. `pilot/pilot_ground_truth.json` — analysis-side reference. List of 30 dicts with `item_id`, `dimension`, `model`, `condition`, `prompt_id`, `llm_judge_median`. Author should not open this file until after scoring.

Also write `pilot/pilot_scores.csv` with columns `item_id, human_score, note` and 30 empty rows for the author to fill.

### Sampling script (helper Claude writes & runs)

Suggested approach:

```python
import json, glob, random, yaml
from collections import defaultdict
from pathlib import Path

random.seed(42)
ROOT = Path('/Users/angelo.leone/Documents/pro-worker-benchmark')

# (1) Find latest result file per (model, variant) containing all 3 layers
files_by_key = {}
for f in sorted(glob.glob(str(ROOT/'dataset_staging/results/*.json'))):
    d = json.loads(Path(f).read_text())
    if not all(k in d for k in ('layer1','layer2','layer3')):
        continue
    files_by_key[(d['model_name'], d['variant'])] = f

# (2) Build pool of L1 (dimension, model, condition, prompt_id, run_idx, response, judge_median)
pool = defaultdict(list)
for (model, variant), f in files_by_key.items():
    d = json.loads(Path(f).read_text())
    for dim, prompts in d['layer1'].items():
        for p in prompts:
            for r_idx, run in enumerate(p['runs']):
                if 'response' not in run or 'judge_scores' not in run:
                    continue
                # judge_median: median of the three judge integer scores
                scores = [int(s['score']) for s in run['judge_scores']]
                judge_median = sorted(scores)[len(scores)//2]
                pool[dim].append({
                    'dimension': dim,
                    'model': model,
                    'condition': variant,
                    'prompt_id': p['prompt_id'],
                    'run_idx': r_idx,
                    'user_prompt': p['prompt'],
                    'response': run['response'],
                    'llm_judge_median': judge_median,
                })

# (3) Stratified sampling per the table above
allocation = {
    'uncertainty_transparency': 6, 'appropriate_reliance': 6,
    'metacognitive_calibration': 4, 'cognitive_forcing': 3,
    'anti_sycophancy': 3, 'complementarity': 2,
    'ethical_surfacing': 2, 'contrastive_explanation': 2,
    'skill_preservation': 1, 'draft_annotation': 1,
}

sampled = []
for dim, n in allocation.items():
    candidates = pool[dim]
    # half baseline, half prompted (round to whole items)
    n_base, n_prompt = n // 2, n - n // 2
    base = [c for c in candidates if c['condition'] == 'baseline']
    prompt = [c for c in candidates if c['condition'] == 'with_system_prompt']
    sampled += random.sample(base, min(n_base, len(base)))
    sampled += random.sample(prompt, min(n_prompt, len(prompt)))

# (4) Write the three output files (items.md, ground_truth.json, scores.csv)
# (helper Claude generates them per the format above)
```

---

## Phase 2 — Author scores blind

The author opens `pilot/pilot_items.md`, reads each item, decides on a 0-3 score using only the rubric anchors shown for that item's dimension, and writes the score into `pilot/pilot_scores.csv`. Optionally a one-line note explaining the call.

Constraints:

- **Do not open `pilot/pilot_ground_truth.json` until after scoring.** This is the entire validity argument — if the author sees the LLM scores, the pilot is worthless.
- Score in one or two sittings. Avoid more. Mental drift between sittings is the main risk.
- Use the rubric strictly. If the response doesn't fit any anchor cleanly, pick the closest and note the ambiguity.
- ~2 minutes per item average. Faster is fine if the call is obvious.

---

## Phase 3 — Analysis

Helper Claude reads `pilot/pilot_scores.csv` and `pilot/pilot_ground_truth.json`, joins on `item_id`, and computes:

```python
import pandas as pd, json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

P = Path('/Users/angelo.leone/Documents/pro-worker-benchmark/pilot')
gt = pd.DataFrame(json.loads((P/'pilot_ground_truth.json').read_text()))
sc = pd.read_csv(P/'pilot_scores.csv')
df = gt.merge(sc, on='item_id')

# Overall
df['exact'] = (df['human_score'] == df['llm_judge_median']).astype(int)
df['adjacent'] = (abs(df['human_score'] - df['llm_judge_median']) <= 1).astype(int)
df['abs_dev'] = abs(df['human_score'] - df['llm_judge_median'])

print(f"N = {len(df)}")
print(f"Exact agreement: {df['exact'].mean()*100:.1f}%")
print(f"Adjacent agreement: {df['adjacent'].mean()*100:.1f}%")
print(f"MAD: {df['abs_dev'].mean():.2f}")
print(f"Quadratic-weighted kappa: {cohen_kappa_score(df['human_score'], df['llm_judge_median'], weights='quadratic'):.3f}")

# UT + AR_d subset
sub = df[df['dimension'].isin(['uncertainty_transparency','appropriate_reliance'])]
print(f"\nUT+AR_d subset (N = {len(sub)})")
print(f"  Exact: {sub['exact'].mean()*100:.1f}%")
print(f"  Adjacent: {sub['adjacent'].mean()*100:.1f}%")

# Per-dimension table
print(df.groupby('dimension').agg(
    n=('item_id','count'),
    exact=('exact','mean'),
    adjacent=('adjacent','mean'),
    mad=('abs_dev','mean'),
).round(2))
```

Write `pilot/pilot_results.md` with the four headline numbers, the per-dimension table, and one line on directional agreement (does the author's score agree on the *direction* of baseline-vs-prompted change at the dimension level?).

---

## Phase 4 — Paper integration

Helper Claude patches the paper. Two locations:

**1. §6.3 (IRR mitigation, main text), append at end of the paragraph:**

> A single-annotator validation pilot ($N = 30$ responses, stratified across all ten Layer-1 dimensions, scored by an author blind to LLM-judge outputs) yields exact agreement of \textbf{X\%}, adjacent agreement of \textbf{Y\%}, and quadratic-weighted Cohen's $\kappa$ = \textbf{Z}; on the two lowest-IRR dimensions (UT, AR\_d, $N = 12$ combined), adjacent agreement is \textbf{W\%}. Full protocol and per-dimension breakdown in Appendix~\ref{app:irr_mitigation} and the released materials.

**2. Appendix C.1 (full IRR mitigation), insert as new (6) consideration:**

> (6)~A single-annotator pilot ($N = 30$, stratified, blind, one of the paper's authors) anchors the LLM-judge median to a human reference: exact agreement \textbf{X\%}, adjacent agreement \textbf{Y\%}, quadratic-weighted $\kappa$ = \textbf{Z}, mean absolute deviation \textbf{D}. The pilot is positioned as a feasibility anchor pending the full multi-annotator study described in Section~\ref{sec:future}; it is reported as a single-annotator estimate with the corresponding caveats.

Substitute the actual numbers. Helper Claude does the `Edit` calls and runs `cp paper_overleaf/*.tex paper/`. Then ask the author to recompile on Overleaf.

**3. Release the pilot.** Drop `pilot/pilot_items.md`, `pilot/pilot_ground_truth.json`, `pilot/pilot_scores.csv`, and `pilot/pilot_results.md` into the supplementary zip rebuild and the HF dataset (under `validation_pilot/`). Update `submission/urls.txt` if you add a separate pointer.

---

## What "good enough" looks like

A defensible pilot at this scope:

- Exact agreement ≥ 50% overall (≥ 60% on high-IRR dimensions, ≥ 30% on UT + AR\_d).
- Adjacent agreement ≥ 80% overall.
- Quadratic-weighted κ ≥ 0.5.
- Mean absolute deviation ≤ 0.7.

These thresholds are typical for ordinal coding pilots of this size. Numbers below those don't invalidate the paper, but warrant an explicit caveat in §6.3 ("the pilot identifies further calibration work, particularly on UT...") and a softening of the IRR mitigation argument.

---

## Risks and mitigations

- **Author bias toward LLM-judge scores.** Mitigation: strict blinding, ground-truth file not opened during scoring. Helper Claude can verify file modification times after Phase 2.
- **Self-evaluation conflict.** The author wrote the rubrics. The pilot tests whether *the author's reading of their own rubric* aligns with the LLM-judge median, not whether *external experts* would. The paper should be explicit about this. The full multi-annotator study in Section 7 is what addresses external validity.
- **N = 30 is small.** Adjacent-agreement and weighted κ have ~±10% margin of error at this N. Report numbers without false precision; use one decimal at most.
- **Dimension imbalance.** Six items per UT and AR\_d gives only ±20% precision per-dimension. Report aggregated UT + AR\_d (N=12) and aggregated high-IRR controls (N=10) instead of per-dimension breakdowns for the lowest-power dimensions.

---

## What to send the helper Claude

Paste this entire file as context, plus:

> Run Phase 1 (sampling). When done, tell me where the items file is and confirm the ground-truth file is not displayed in chat. I'll do Phase 2 in my editor and ping you when the scores CSV is filled. Then run Phase 3 and Phase 4.

The helper does not need the LLM-judge scores in chat — only the protocol and the file paths.
