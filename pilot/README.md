# Validation Pilot — Pro-Worker AI Benchmark

This directory packages the supplementary pilot referenced in §6.3 and Appendix C.1 of the paper.
Two analyses are included: a single-annotator human pilot (N=30) and a Claude-as-fourth-judge
inter-LLM-judge consistency check (N=500). Both anchor the existing three-judge panel without
substituting for the multi-annotator study described in Section 7 (Future Work).

## Files

### Human pilot (N=30)
- `sample_pilot.py` — stratified sampler (seed=42)
- `pilot_items.md` — author-facing scoring sheet, rubric anchors per item, no LLM scores leaked
- `pilot_scores.csv` — author scores plus optional one-line notes
- `pilot_ground_truth.json` — sample identifiers and LLM-judge medians (analysis-side only;
  the annotator did not view this file during scoring)
- `pilot_results.md` — exact / adjacent / MAD / quadratic-weighted κ, headline and per-dimension

### Fourth judge (N=500)
- `sample_fourth_judge.py` — stratified sampler (seed=4242)
- `run_fourth_judge.py` — invokes Claude Sonnet 4.6 via the `claude` CLI under the same judge
  system prompt, rubric, and few-shot protocol as the panel
- `fourth_judge_items.jsonl` — sampled items with prompts, responses, and panel scores
- `fourth_judge_scores.jsonl` — Claude's scores plus reasoning and evidence
- `fourth_judge_results.md` — agreement metrics overall, per-dimension, per-model, per-condition

### Shared
- `analyze.py` — joins scores against panel medians and computes all reported metrics

## Reproducing

The samples are reproducible from the raw result files in `dataset_staging/results/`:

```
python pilot/sample_pilot.py            # → pilot_items.md, pilot_ground_truth.json, pilot_scores.csv
python pilot/sample_fourth_judge.py     # → fourth_judge_items.jsonl
python pilot/run_fourth_judge.py        # → fourth_judge_scores.jsonl (requires `claude` CLI auth)
python pilot/analyze.py both            # → pilot_results.md, fourth_judge_results.md
```

Both samplers use a fixed seed; the sampling is therefore deterministic given the released
result files. The fourth-judge run is non-deterministic (Claude API at temperature 0 is
near-deterministic but not bitwise-reproducible across runs); aggregate agreement metrics
should be stable to ±1-2 points across replications.

## Headline numbers (also in the two `*_results.md` files)

| Metric | Human pilot (N=30) | Fourth judge (N=500) |
|---|---|---|
| Exact agreement | 56.7% | 66.8% |
| Adjacent agreement (±1) | 86.7% | 96.8% |
| Mean absolute deviation | 0.60 | 0.36 |
| Quadratic-weighted Cohen's κ | 0.702 | 0.840 |

The fourth judge independently identifies metacognitive calibration (κ=0.51) and uncertainty
transparency (κ=0.58) as its lowest-agreement dimensions, mirroring the panel's IRR pattern.

## Methodological positioning

- The human pilot tests whether the rubric author's reading of the rubric agrees with the
  panel median. External rubric validity (whether independent expert annotators reach the
  same scores) remains the target of the future multi-annotator study.
- The fourth-judge analysis is an inter-LLM-judge consistency check across model families,
  using the only closed-weight model in the entire judge or evaluation stack. It does not
  measure human agreement.
- Together, the two analyses shift the §6.3 IRR mitigation from a structural argument to
  a structural argument plus two empirical anchors.
