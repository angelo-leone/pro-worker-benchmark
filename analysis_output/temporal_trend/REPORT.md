# Temporal-trend analysis: are LLMs becoming more automating, less augmenting?

**Question.** Across model release dates, does baseline pro-worker behavior decline (i.e., do newer models default to "just answer" rather than "engage the human in thinking")?

**Short answer.** The data we have *cannot* answer this cleanly. A naive cross-cohort regression shows a strong, significant decline (-1.06 PWI-points/month, p=0.0001), but the effect is confounded with a judging-protocol change between the v1 (Feb 2026) and v2 (Apr 2026) data collection rounds. Within either protocol cohort the slope is not significant. To answer the question properly we need to re-judge the v1 responses under the v2 protocol; that response text is on disk (~360 prompts × 3 models) and the re-judge would cost on the order of $5–10 and a few hours of judge time.

---

## What was done

10 baseline runs (no system prompt) across 10 open-weight models from 8 families, spanning 2024-07 to 2026-02. Restricted to a fair-comparison subset:

- **6 dimensions** present in both v1 and v2 rubric files: cognitive_forcing, complementarity, contrastive_explanation, draft_annotation, skill_preservation, uncertainty_transparency.
- **15 prompt IDs** per dimension (`*_01..*_15`) shared by both prompt sets. Prompt text verified identical for shared IDs.
- **Baseline variant only.** The "automating vs augmenting" question is about default behavior; the system-prompt variant tests instructability, not defaults.
- **PWI-6** = production weights restricted to those 6 dimensions, renormalized to 1, scaled to 0–100. Comparable across cohorts on a relative scale, but absolute values are not comparable to the paper's PWI (which uses 11 dimensions).

Code: `analyze_temporal_trend.py` in this directory. Outputs: 4 figures (PNG + PDF), 3 CSVs, this report.

## Headline (naive) result

| metric | value |
|---|---|
| n models | 10 |
| slope | **−1.06 PWI-points / month** |
| 95% CI | [−1.38, −0.74] |
| Pearson r | −0.94 |
| p (two-sided) | 0.0001 |

Per-dimension slopes (raw 0–3 score per month):

| dimension | slope/mo | r | p |
|---|---:|---:|---:|
| complementarity | **−0.065** | −0.95 | <0.0001 |
| uncertainty_transparency | **−0.057** | −0.75 | 0.013 |
| draft_annotation | **−0.046** | −0.76 | 0.011 |
| cognitive_forcing | **−0.019** | −0.72 | 0.019 |
| contrastive_explanation | −0.012 | −0.30 | 0.40 (n.s.) |
| skill_preservation | +0.004 | +0.12 | 0.74 (n.s.) |

If taken at face value: 4 of 6 augmentation-relevant dimensions decline significantly across releases; complementarity collapses near-monotonically (r = −0.95).

## Why I do not trust the headline

The Feb 2026 (v1) data was scored by a **single judge**; the Apr 2026 (v2) data was scored by a **3-judge median panel**. The two cohorts also happen to be temporally separated. Stratifying:

| cohort | n | release range | mean PWI-6 | within-cohort slope/mo | p |
|---|---:|---|---:|---:|---:|
| v1 (single judge) | 3 | 2024-07 → 2025-03 | **38.7** | −0.78 | 0.43 |
| v2 (3-judge median) | 7 | 2025-08 → 2026-02 | **24.3** | −0.28 | 0.57 |

Within each cohort the slope is not significant. The between-cohort level difference is significant (Welch t = +5.81, p = 0.016).

The pattern is a **level shift, not a slope**: the older models all sit ~38 PWI-6 and the newer models all sit ~24 PWI-6, with no model in between. That is the signature of a methodology change, not a smooth temporal drift.

Further evidence the level shift is at least partly methodological:

- **skill_preservation does not drop** (+0.004/mo, p=0.74). If the judging protocol uniformly inflated v1 scores, you would expect every dimension to show the same drop. It does not.
- **complementarity drops 4×** (1.16 → 0.27 raw mean). This is the most suspicious magnitude. A 4× collapse in the same dimension across every model from every family in a 6-month gap is more parsimoniously explained by the panel-vs-single judge being differently calibrated on that dimension than by every model vendor independently changing their training in that direction.
- **cognitive_forcing's drop is plausible**: even in v1 most models scored near floor (0.13–0.50 raw, on a 0–3 scale). Default models barely ever cognitive-force. The v2 floor (0.00–0.28) is a small additional compression that could reflect either more confident "just answer" defaults or a stricter panel.

I cannot disentangle real model drift from judge drift with this data alone.

## What we'd need to actually answer the question

The v1 response text (~360 prompts × 3 models) is on disk in `results/openrouter_*_baseline_*.json`. Re-judging those responses with the current 3-judge panel would isolate the model effect from the judge effect. Rough cost estimate based on `docs/cost_estimate.md` ratios: ~270 prompts × 3 judges × ~1k tokens each ≈ 800k judge tokens, well under $10 on Vultr at current rates.

If after re-judging the v1 models still score ~38 PWI-6 and the v2 models still score ~24, the trend is real and worth a follow-up note. If the v1 cohort drops to v2 levels under the same panel, the apparent trend is a methodology artifact.

A second avenue, which avoids new API calls: pick a single model that exists in both eras under matching evaluation (e.g., a Llama or Qwen baseline already judged by both protocols). We do not currently have that, but if any of the v1 models can be cheaply re-run through the v2 pipeline end-to-end, that doubles as a calibration probe.

## Things I deliberately did not do

- I did not try to "explain away" the level shift with confounds I cannot test (size, family mix, instruction-tuning era). Those are real but downstream of the judge-confound question.
- I did not extend the trend line to predict future PWI. With n=10 and a confound this large, point predictions are not credible.
- I did not touch the paper. The temporal-trend question is not what the paper claims; the paper reports a cross-section of 7 models all run under the v2 protocol, which is internally consistent.

## Files in this directory

- `analyze_temporal_trend.py` — the analysis script; rerun any time.
- `pwi6_by_model.csv` — 10 models × {release date, PWI-6, per-dim means, n}.
- `per_dim_trend.csv` — per-dimension OLS slope, r, p.
- `all_scores_long.csv` — every individual score used (model × dimension × score).
- `fig_pwi6_vs_date.{png,pdf}` — naive trend plot with regression line.
- `fig_per_dim_trend.{png,pdf}` — small multiples per dimension.
- `fig_bucket.{png,pdf}` — half-year cohort means (more robust to date error).
- `fig_protocol_confound.{png,pdf}` — the stratified plot showing the v1/v2 level break.

## Release dates used

Verified from naming or public release: Llama 3.1 70B (2024-07-23), Qwen2.5 72B (2024-09-19), Mistral Small 3.1 (2025-03-17), GPT-oss 120B (2025-08-05), Devstral-2 123B (2025-12, from the `2512` tag in the model id). Estimated from family naming conventions (may be off by ±1 quarter): Gemma 4 31B, Nemotron-Cascade 30B, DeepSeek V3.2, Qwen3.5 27B, GLM 5.1. Date errors of ±1 quarter do not change any of the conclusions above; they would shift point positions but not the cross-cohort discontinuity.
