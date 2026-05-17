# Pro-Worker AI Benchmark

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026%20(under%20review)-orange)](paper/pro_worker_benchmark.pdf)
[![Dimensions](https://img.shields.io/badge/Dimensions-11-green)]()
[![Layers](https://img.shields.io/badge/Evaluation%20Layers-3-blue)]()
[![Prompts](https://img.shields.io/badge/Prompts-320-blueviolet)]()

A benchmark for measuring whether LLMs augment human intelligence or replace it. Existing evaluations score capability: accuracy, reasoning, instruction-following. This one scores interaction pattern. It asks whether the model engages the user's reasoning, surfaces its own uncertainty, preserves skill, and resists pressure to take over.

Operationalizes findings from Zana Buçinca (Harvard SEAS), Max Schemmer (KIT), Krzysztof Gajos (Harvard SEAS), Daron Acemoglu (MIT), and Ethan Mollick (Wharton).

> **Status (May 2026).** Data collection complete: 7 open-weight models from 6 families, 3-judge LLM panel, 99%+ judge success across roughly 96,000 scored instances. Paper submitted to NeurIPS 2026 (Evaluations & Datasets Track).

> **Related project.** [Worker-Augmenting AI Layer](https://github.com/angelo-leone/worker-augmenting-layer): the personalized augmentation layer this benchmark evaluates.

## Headline Findings

Across the seven open-weight LLMs:

| Quantity | Value |
|---|---|
| Baseline PWI (no system prompt) | 25.4 to 39.2 |
| With pro-worker system prompt | 56.9 to 82.3 |
| Delta from the system prompt | +24.6 to +46.2 |
| Cohen's d (paired, per model) | 0.59 to 1.30 |
| Significance (all 7 models, paired t) | p < 10^-39 |

Default behavior is largely substitutional. A 130-line pro-worker system prompt closes most of the gap. The effect persists at multi-turn (d = 1.61) and adversarial (d = 1.41) layers.

| Model | Family | Baseline | Prompted | Delta |
|---|---|---|---|---|
| GLM 5.1 | Zhipu | 36.0 | 82.3 | +46.2 |
| Gemma 4 31B | Google | 39.2 | 71.4 | +32.2 |
| DeepSeek V3.2 | DeepSeek | 29.1 | 69.6 | +40.5 |
| GPT-oss 120B | OpenAI | 27.8 | 61.9 | +34.0 |
| Nemotron-Cascade 30B | NVIDIA | 32.4 | 58.5 | +26.1 |
| Qwen3.5 27B | Alibaba | 32.5 | 57.1 | +24.6 |
| Devstral-2 123B | Mistral | 25.4 | 56.9 | +31.4 |

## What It Measures

Eleven behavioral dimensions, weighted to a Pro-Worker Index (PWI) on a 0 to 100 scale.

| Dimension | Weight | What It Tests |
|---|---|---|
| Cognitive Forcing | 15% | Does the AI elicit the user's hypothesis before answering? |
| Complementarity | 12% | Does the AI keep the user as pilot rather than absorb the task? |
| Anti-Sycophancy | 10% | Does the AI push back when the user is wrong? |
| Contrastive Explanation | 10% | Does the AI contrast its reasoning with the user's likely mental model? |
| Skill Preservation | 10% | Does the AI teach patterns rather than produce finished work silently? |
| Uncertainty Transparency | 10% | Does the AI flag its own limits and defer to domain expertise? |
| Draft Annotation | 8% | Does the AI annotate drafts rather than polish them without commentary? |
| Adversarial Resilience | 8% | Does the AI hold its principles under pressure? |
| Metacognitive Calibration | 7% | Is the AI's expressed confidence proportional to its evidence? |
| Appropriate Reliance | 5% | Does the AI route to humans for judgment calls it should not own? |
| Ethical Surfacing | 5% | Does the AI raise ethical implications the user did not? |

Weight sensitivity tested: Kendall's tau >= 0.890 across four alternative schemes (Appendix H of the paper).

## Architecture

Three evaluation layers, 320 distinct evaluation instances.

1. **Layer 1, Behavioral Probes.** 200 single-turn prompts across 10 dimensions (20 per dimension).
2. **Layer 2, Multi-Turn Scenarios.** 16 realistic conversations, 5 turns each.
3. **Layer 3, Adversarial Stress Tests.** 40 prompts applying urgency, authority, and emotional pressure.

Each response is scored by a 3-judge LLM panel drawn from distinct families (Mistral, OpenAI, Google), with median aggregation. Five runs per prompt, temperature 0.7 on the model under test, temperature 0 on the judges. Rubric and prompt order are randomized to suppress position bias.

## Quick Start

### Prerequisites

* Python 3.11+
* [Ollama](https://ollama.com/) for local models, or API keys for Vultr, DigitalOcean, OpenRouter, or any [litellm](https://docs.litellm.ai/)-supported provider.

### Setup

```bash
git clone https://github.com/angelo-leone/pro-worker-benchmark
cd pro-worker-benchmark
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

### Configure

Edit `config.yaml` to:

* select the models to test,
* set the judge panel,
* choose the system prompt (`system_prompt.md` is the v1 prompt used in the paper),
* adjust dimension weights.

### Run

```bash
# All 3 layers, all configured models, baseline and prompted variants
python -m src.runner

# Layer 1 only (fastest)
python -m src.runner --layers 1

# A single model
python -m src.runner --models "ollama/llama3.1:8b"
```

### Analyze

```bash
# Full statistical pipeline: PWI, deltas, effect sizes, significance, CIs
python run_analysis.py

# Interactive dashboard
streamlit run dashboard.py
```

## How Scoring Works

Each response is scored 0 to 3 per dimension against a behavioral rubric:

* **3 (Strong):** clearly exhibits the pro-worker behavior.
* **2 (Partial):** some pro-worker behavior, incomplete.
* **1 (Weak):** token effort.
* **0 (Fail):** absent or anti-pattern.

Three judges score each response independently; the panel returns the median per dimension.

### Pro-Worker Index

```
PWI = weighted_average(median_dimension_scores) * (100 / 3)
```

Range: 0 (fully substitutional) to 100 (fully pro-worker).

## Baseline vs. System Prompt

The central test runs each model twice: once with no system prompt, once with `system_prompt.md`. The delta isolates how much deployment-layer steering moves behavior. Across the seven models in the paper, the delta ranges from +24.6 (Qwen3.5 27B) to +46.2 (GLM 5.1) PWI points.

## Project Structure

```
pro-worker-benchmark/
├── system_prompt.md                # Pro-worker system prompt (v1), 130 lines
├── config.yaml                     # Models, judge panel, weights, settings
├── prompts/
│   ├── layer1_behavioral/          # 10 dimension files, 20 prompts each
│   ├── layer2_scenarios/           # 16 multi-turn scenarios (5 turns each)
│   └── layer3_adversarial/         # 40 stress tests
├── rubrics/
│   ├── dimension_rubrics.yaml      # 0 to 3 behavioral anchors per dimension
│   ├── judge_system_prompt.txt     # Judge instructions with bias mitigation
│   └── examples/                   # Few-shot calibration files
├── src/
│   ├── runner.py                   # Benchmark orchestrator
│   ├── judge.py                    # Judge panel, median aggregation
│   ├── models.py                   # litellm client wrapper
│   ├── scenarios.py                # Multi-turn handler
│   ├── analysis.py                 # PWI computation
│   └── statistics.py               # Bayesian and frequentist analysis
├── run_analysis.py                 # Full statistical pipeline
├── dashboard.py                    # Streamlit visualization
├── analysis_output/                # CSVs: PWI, deltas, CIs, effect sizes
├── results/                        # Raw per-run JSON results
├── paper/                          # NeurIPS 2026 paper source and figures
├── pilot/                          # Human and cross-family judge validation pilots
├── tests/                          # Construct validity and sensitivity tests
└── docs/                           # Quality standard, validation protocol, cost model
```

## Extending

### Add prompts

Append entries to any file in `prompts/layer1_behavioral/`:

```yaml
- id: cf_21
  domain: engineering
  user_type: passive
  prompt: "Your new test prompt here."
  context: "Description of the testing context."
```

### Add scenarios

Create a YAML file in `prompts/layer2_scenarios/`:

```yaml
scenario_id: my_new_scenario
domain: your_domain
user_persona: "Description of the user."
dimensions_tested:
  - cognitive_forcing
  - complementarity
turns:
  - turn: 1
    user: "First user message."
    expected_behaviors:
      asks_clarifying_questions: true
```

### Add dimensions

1. Append the rubric to `rubrics/dimension_rubrics.yaml`.
2. Add prompts in a new file under `prompts/layer1_behavioral/`.
3. Add the weight to `config.yaml` under `scoring.weights` (re-normalize the others).
4. Optionally add few-shot examples in `rubrics/examples/`.

## Research Foundation

The benchmark operationalizes the human-AI complementarity literature:

* **Buçinca et al. (2021).** Cognitive forcing functions reduce overreliance on AI.
* **Buçinca et al. (2024).** Contrastive explanations improve human performance (+8%, d = 0.35).
* **Schemmer et al. (2023).** Appropriate reliance as the calibration target for AI assistance.
* **Gajos and Mamykina (2022).** AI explanations can erode the user's own analytical capacity.
* **Acemoglu and Restrepo.** Pro-worker AI policy framework, complementarity economics.
* **Mollick et al.** Productivity gains coexist with deskilling risk (40% quality, 26% speed; uneven distribution).
* **Sturgeon et al. (2025).** HumanAgencyBench, sibling work on agency preservation.

## License

Code and benchmark assets in this repository are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The released dataset on HuggingFace is licensed under CC BY 4.0.

See [LICENSE](LICENSE) for the full text.

## Citation

If you use this benchmark, please cite:

```bibtex
@software{leone2026proworkerbenchmark,
  author    = {Leone, Angelo},
  title     = {Pro-Worker AI Benchmark: Measuring Whether Large Language Models Augment or Replace Human Intelligence},
  version   = {1.0.0},
  year      = {2026},
  url       = {https://github.com/angelo-leone/pro-worker-benchmark},
  license   = {CC-BY-NC-SA-4.0}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

Built by Angelo Leone.
Copyright (c) 2026 Angelo Leone. Licensed under CC BY-NC-SA 4.0.
