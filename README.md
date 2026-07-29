# Pro-Worker AI Benchmark

[![Code License: MIT](https://img.shields.io/badge/Code-MIT-lightgrey.svg)](https://opensource.org/license/mit)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026%20(under%20review)-orange)](paper/pro_worker_benchmark.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/angelo-leone/pro-worker-ai-benchmark)
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
| Baseline PWI (no system prompt) | 25.8 to 40.8 |
| With pro-worker system prompt | 58.2 to 84.7 |
| Delta from the system prompt | +24.1 to +44.0 |
| d, pooled over dimensions (per model) | 0.59 to 1.30 |
| Significance (all 7 models, paired t) | p < 10^-39 |

Default behavior is largely substitutional. A 130-line pro-worker system prompt closes most of the gap. The effect persists at multi-turn (d = 1.61) and adversarial (d = 1.41) layers.

| Model | Family | Baseline | Prompted | Delta |
|---|---|---|---|---|
| GLM 5.1 | Zhipu | 40.7 | 84.7 | +44.0 |
| Gemma 4 31B | Google | 40.8 | 77.2 | +36.4 |
| DeepSeek V3.2 | DeepSeek | 29.8 | 72.2 | +42.4 |
| GPT-oss 120B | OpenAI | 28.9 | 64.1 | +35.2 |
| Nemotron-Cascade 30B | NVIDIA | 33.2 | 60.1 | +26.9 |
| Devstral-2 123B | Mistral | 25.8 | 60.0 | +34.2 |
| Qwen3.5 27B | Alibaba | 34.1 | 58.2 | +24.1 |

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

Weight sensitivity tested: Kendall's tau ranges 0.619 to 0.810 across three alternative schemes (Appendix H of the paper). Rank order among middle-placed models is weight-dependent; the substitution-versus-augmentation contrast and the prompt effect are not.

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

The central test runs each model twice: once with no system prompt, once with `system_prompt.md`. The delta isolates how much deployment-layer steering moves behavior. Across the seven models in the paper, the delta ranges from +24.1 (Qwen3.5 27B) to +44.0 (GLM 5.1) PWI points.

## For AI Labs and External Evaluators

The full v2.0 release is hosted on HuggingFace: [angelo-leone/pro-worker-ai-benchmark](https://huggingface.co/datasets/angelo-leone/pro-worker-ai-benchmark). It contains all 320 prompts, 11 dimension rubrics, few-shot calibration files, the v1 pro-worker system prompt, and the full per-run JSON results from the seven models reported in the paper (~96,000 scored instances, ~320MB). This code repository holds the code and derived analysis; the raw per-run results live on HuggingFace rather than in git.

To evaluate a new model:

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="angelo-leone/pro-worker-ai-benchmark",
    repo_type="dataset",
)
```

Then point the runner at any litellm-supported endpoint. Step-by-step provider examples (OpenAI, Anthropic, Bedrock, Vertex, OpenRouter, local Ollama, custom vLLM/TGI) live in [docs/providers.md](docs/providers.md), including per-provider cost estimates for a full run.

The judge panel is configurable: replace any of the three judges in `config.yaml`, change `judge_aggregation` to `mean` or `min`, or run a single judge for cost. Five runs per prompt are recommended for the headline numbers; a single-run dry pass costs roughly 1/5 and still reproduces dimension ordering.

Reproducibility:

* Pin to the `v2.0.0` git tag of this repo and the matching HuggingFace dataset revision.
* The released v2.0 result JSONs are byte-stable; re-running `python run_analysis.py` on them reproduces every figure and table in the paper exactly.
* Weight sensitivity is documented (Kendall's tau 0.619 to 0.810 across three alternative schemes); custom dimension weights can be applied without re-running inference.
* The validation pilot bundle (single-annotator N=30, cross-family fourth-judge N=500) lives in `validation_pilot/` on the HF dataset and is the recommended starting point for any human-validation extension.

To publish results: see [LEADERBOARD.md](LEADERBOARD.md) for the current standings and the PR template for adding a new model.

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
├── results/                        # Raw per-run JSON results (generated locally when you run the benchmark; the released v2.0 set lives on HuggingFace, not in this repo)
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

The repository is dual-licensed to match the de facto standard for ML benchmarks:

* **Source code** (`src/`, `run_analysis.py`, `dashboard.py`, `upload_to_hf.py`, tests, top-level Python scripts) is released under the [MIT License](LICENSE). Commercial use is permitted; attribution is the only requirement.
* **Data assets** (`prompts/`, `rubrics/`, `system_prompt.md`, `results/`, `analysis_output/`, and the HuggingFace dataset mirror) are released under [CC BY 4.0](LICENSE-DATA). Commercial use is permitted; attribution is the only requirement.
* **Paper PDF and LaTeX source** (`paper/`) remain subject to the publication venue's terms at acceptance.

In practice: AI labs, research groups, and external evaluators can run the benchmark, fork the code, modify the rubrics, publish derivative work, and integrate results into commercial pipelines, provided the author and the benchmark are credited.

Attribution string for the data:
> Angelo Leone, *Pro-Worker AI Benchmark v2.0*, 2026. https://huggingface.co/datasets/angelo-leone/pro-worker-ai-benchmark

See [LICENSE](LICENSE), [LICENSE-DATA](LICENSE-DATA), and [COPYRIGHT](COPYRIGHT) for the full text.

## Citation

If you use this benchmark, please cite:

```bibtex
@software{leone2026proworkerbenchmark,
  author    = {Leone, Angelo},
  title     = {Pro-Worker AI Benchmark: Measuring Whether Large Language Models Augment or Replace Human Intelligence},
  version   = {2.0.0},
  year      = {2026},
  url       = {https://github.com/angelo-leone/pro-worker-benchmark},
  license   = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

Built by Angelo Leone.
Copyright (c) 2026 Angelo Leone. Source code under the MIT License; data assets under CC BY 4.0.
