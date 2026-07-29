# Claude-as-Fourth-Judge Agreement (vs existing 3-judge panel median)

Methodology: Claude Sonnet 4.6 was given the same judge system prompt, rubric, and few-shot examples as the existing panel (Devstral-2 123B, GPT-oss 120B, Gemma 4 31B), then scored a stratified N=500 sample of Layer-1 instances. This is *not* human validation; it is an inter-LLM-judge consistency check across model families.

Items scored: **500**  (skipped 0)


## Headline numbers

- Exact agreement: **66.8%**
- Adjacent agreement (±1): **96.8%**
- Mean absolute deviation: **0.36**
- Quadratic-weighted Cohen's κ: **0.840**

## Per-dimension

| Dimension                 | N  | Exact % | Adjacent % | MAD  | QW-κ  |
|---------------------------|----|---------|------------|------|-------|
| anti_sycophancy           | 50 | 84.0%   | 100.0%     | 0.16 | 0.927 |
| appropriate_reliance      | 50 | 70.0%   | 98.0%      | 0.32 | 0.778 |
| cognitive_forcing         | 50 | 72.0%   | 100.0%     | 0.28 | 0.877 |
| complementarity           | 50 | 70.0%   | 98.0%      | 0.32 | 0.880 |
| contrastive_explanation   | 50 | 64.0%   | 100.0%     | 0.36 | 0.755 |
| draft_annotation          | 50 | 88.0%   | 100.0%     | 0.12 | 0.942 |
| ethical_surfacing         | 50 | 60.0%   | 98.0%      | 0.42 | 0.726 |
| metacognitive_calibration | 50 | 48.0%   | 84.0%      | 0.68 | 0.505 |
| skill_preservation        | 50 | 64.0%   | 98.0%      | 0.38 | 0.644 |
| uncertainty_transparency  | 50 | 48.0%   | 92.0%      | 0.60 | 0.583 |


## Per-model

| Model                | N  | Exact % | Adjacent % | MAD  | QW-κ  |
|----------------------|----|---------|------------|------|-------|
| DeepSeek V3.2        | 62 | 69.4%   | 100.0%     | 0.31 | 0.890 |
| Devstral-2 123B      | 82 | 72.0%   | 100.0%     | 0.28 | 0.884 |
| GLM 5.1              | 79 | 72.2%   | 94.9%      | 0.33 | 0.846 |
| GPT-oss 120B         | 70 | 57.1%   | 98.6%      | 0.44 | 0.808 |
| Gemma 4 31B          | 63 | 63.5%   | 93.7%      | 0.43 | 0.807 |
| Nemotron-Cascade 30B | 72 | 66.7%   | 94.4%      | 0.39 | 0.792 |
| Qwen3.5 27B          | 72 | 65.3%   | 95.8%      | 0.39 | 0.773 |


## Per-condition

| Condition          | N   | Exact % | Adjacent % | MAD  | QW-κ  |
|--------------------|-----|---------|------------|------|-------|
| baseline           | 250 | 72.8%   | 98.0%      | 0.29 | 0.826 |
| with_system_prompt | 250 | 60.8%   | 95.6%      | 0.44 | 0.793 |
