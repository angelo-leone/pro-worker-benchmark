# Judge availability

The published scores use a three-judge panel served by Vultr: Devstral-2 123B, GPT-oss 120B and Gemma 4 31B (Intel int4 AutoRound build), median aggregation, temperature 0.

Hosted open-weight models get retired. As of 30 October 2026:

- Vultr's public model list no longer includes any of the three judges.
- DigitalOcean, the backup provider, retires Gemma 4 31B on 9 November 2026.
- OpenRouter serves all three (`mistralai/devstral-2512`, `openai/gpt-oss-120b`, `google/gemma-4-31b-it`).

## Running with the OpenRouter panel

```bash
export OPENROUTER_API_KEY=...
export PWB_JUDGE_PANEL=openrouter
python run_parallel.py ...
```

The panel is defined under `judge_panels.openrouter` in `config.yaml`. Leaving `PWB_JUDGE_PANEL` unset uses the published panel.

## Before mixing new scores with published ones

OpenRouter's Gemma 4 31B is not the int4 AutoRound build, and providers may serve different quantisations of the other two. Scores from the new panel are not automatically comparable with the published ones. Before using them together:

1. Take a stratified sample of at least 100 published responses (both conditions, all dimensions) from the Hugging Face dataset.
2. Re-score them with `PWB_JUDGE_PANEL=openrouter`.
3. Report quadratic-weighted kappa between the old and new panel medians, per dimension and overall, and the shift in each model's PWI.
4. If agreement is below 0.70 on a dimension, treat that dimension's new scores as a separate run, the same rule the leaderboard applies to non-comparable submissions.
