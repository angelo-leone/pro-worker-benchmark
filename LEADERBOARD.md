# Pro-Worker AI Benchmark Leaderboard

Current as of 2026-05-17 (v2.0). Pro-Worker Index (PWI) is a 0 to 100 weighted aggregate across 11 behavioral dimensions, scored by a 3-judge LLM panel with median aggregation across 5 runs per prompt. Higher is better.

## Results (seven open-weight models, six families)

| Rank (prompted) | Model | Family | Size | Baseline PWI | Prompted PWI | Delta | Cohen's d |
|---|---|---|---|---|---|---|---|
| 1 | GLM 5.1 | Zhipu | Large | 36.0 | 82.3 | +46.2 | 1.30 |
| 2 | Gemma 4 31B | Google | 31B | 39.2 | 71.4 | +32.2 | 0.59 |
| 3 | DeepSeek V3.2 | DeepSeek | Large | 29.1 | 69.6 | +40.5 | 0.95 |
| 4 | GPT-oss 120B | OpenAI | 120B | 27.8 | 61.9 | +34.0 | 0.63 |
| 5 | Nemotron-Cascade 30B | NVIDIA | 30B | 32.4 | 58.5 | +26.1 | 0.79 |
| 6 | Qwen3.5 27B | Alibaba | 27B | 32.5 | 57.1 | +24.6 | 0.75 |
| 7 | Devstral-2 123B | Mistral | 123B | 25.4 | 56.9 | +31.4 | 0.85 |

Confidence intervals (95% bootstrap) and per-dimension scores live in `analysis_output/pwi_scores.csv` and `analysis_output/dimension_scores.csv`. Raw per-run JSON is in the HuggingFace dataset under `results/`.

## How to submit a new model

We accept pull requests adding new entries. Submissions are validated against the published v2.0 prompts, rubrics, system prompt, and judge panel for direct comparability with the table above.

### Required artifacts

Open a PR that adds:

1. A new row in the table above with your model's Baseline PWI, Prompted PWI, Delta, and Cohen's d, plus a citation/URL for the model.
2. The raw per-run JSON results (both `baseline` and `with_system_prompt` variants) committed to `results/` and uploaded to the HF dataset under `results/`. File names follow the existing convention: `{provider}_{vendor}_{model-id}_{variant}_{YYYYMMDD_HHMMSS}.json`.
3. A short `submissions/{model_id}.md` describing run configuration: provider, API endpoint, temperatures used, judge panel (if non-default), random seed, total cost.

### Required configuration

To remain on the comparable leaderboard:

* **Prompts:** the exact 320 instances published in v2.0 (no modifications, additions, or removals).
* **System prompt for the prompted variant:** the exact text in `system_prompt.md` from the v2.0 tag, applied as the system message.
* **Judge panel:** the three-model panel in `config.yaml` (Devstral-2 123B, GPT-oss 120B, Gemma 4 31B), median aggregation, temperature 0.0.
* **Runs per prompt:** 5, at model temperature 0.7.
* **Pin the commit:** use the `v2.0.0` git tag of this repository and the corresponding HF dataset revision.

Submissions that deviate (alternate judge, single-run, custom prompt set, modified rubrics) are still welcome but land in a separate `## Non-comparable runs` section with the deviation called out.

### Submission template

Use this PR description:

```
### Model: {full model identifier}

| Field | Value |
|---|---|
| Provider | OpenAI / Anthropic / Vertex / Bedrock / Together / Vultr / local Ollama / ... |
| API endpoint | {URL or "official SDK"} |
| Model temperature | 0.7 |
| Judge panel | default (Devstral-2 123B, GPT-oss 120B, Gemma 4 31B), median |
| Runs per prompt | 5 |
| Total scored instances | 2 variants x 320 prompts x 5 runs x 3 judges = 9,600 |
| Total API cost | $X (model) + $Y (judges) = $Z |
| Wall-clock time | H hours |
| Repo commit pinned | v2.0.0 |
| HF dataset revision pinned | {commit sha} |
| Anything non-default | none / {brief note} |

### Headline numbers

| | Baseline PWI | Prompted PWI | Delta |
|---|---|---|---|
| Mean (5 runs) | {x} | {y} | +{z} |
| 95% CI (bootstrap, B=10000) | [{a}, {b}] | [{c}, {d}] | [{e}, {f}] |
| Cohen's d (paired) | {d} |
| p-value (paired t) | {p} |

### Notes

{Any anomalies: judge parse failures, refusals, rate-limit retries, etc.}
```

## What "non-comparable" means in practice

Some labs cannot use the default judge panel (model-policy restrictions, judge unavailability, budget). Submitting with a different judge panel is fine; the result lives in a separate section and we report the panel composition so readers can interpret the delta against the baseline numbers carefully. The same applies to single-run submissions, custom temperature, or a custom system prompt variant.

## Contact

Open an issue or PR at https://github.com/angelo-leone/pro-worker-benchmark. For sensitive submissions (unreleased models, private API access), email angelo.leone1204@gmail.com to coordinate.
