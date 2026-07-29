# Reproducibility Checklist

Following the NeurIPS 2026 reproducibility guidelines.

---

## Claims and Scope

- [x] **The paper's main claims clearly stated in the abstract and introduction.**
  Claims: (1) baseline LLMs overwhelmingly substitutional (PWI 25-39); (2) system prompts yield large, statistically significant improvements (mean +33.6 PWI, all d ≥ 0.44); (3) dimensions vary in malleability; (4) single-turn/multi-turn performance partially dissociated; (5) adversarial resilience independent of general PWI.

- [x] **Claims are supported by the experimental results.**
  Each claim is backed by specific tables and figures with confidence intervals.

- [x] **Clear scope and limitations stated.**
  English-only, 7 open-weight models evaluated, judge-candidate overlap documented, construct overlap disclosed.

---

## Experimental Setup

- [x] **Model details provided** (Section 4.1, Table 1):
  - 7 models: DeepSeek V3.2, Devstral-2 123B, Gemma 4 31B, GPT-oss 120B, Nemotron-Cascade 30B, Qwen3.5 27B, GLM 5.1
  - All open-weight, accessed via Vultr Serverless Inference API
  - Temperature 0.7, max_tokens 8192

- [x] **Judge panel details provided** (Section 4.3):
  - 3 judges from different families (Devstral-2 123B, GPT-oss 120B, Gemma 4 31B)
  - Temperature 0.0, max_tokens 1024
  - Median aggregation

- [x] **Prompts publicly available.**
  All 320 prompts released in `prompts/` directory.

- [x] **Rubrics publicly available.**
  All 11 dimension rubrics released in `rubrics/dimension_rubrics.yaml`.

- [x] **System prompt publicly available.**
  Released in `system_prompt.md` (130 lines).

- [x] **Judge system prompt publicly available.**
  Released in `rubrics/judge_system_prompt.txt`.

---

## Statistical Methodology

- [x] **Multiple runs to estimate variance.**
  5 runs per prompt for Layer 1 and Layer 3; single-run Layer 2 (by design, multi-turn trajectories).

- [x] **Confidence intervals reported.**
  95% Bayesian HDI (Dirichlet-Multinomial posterior) for dimension means.
  95% bootstrap CI (10,000 resamples) for composite PWI scores.

- [x] **Effect sizes reported.**
  Cohen's d for all system-prompt comparisons, per model and per dimension.

- [x] **Significance testing.**
  Paired Wilcoxon signed-rank (per dimension) and paired t-tests (per model).
  Bonferroni correction for multiple dimensions.

- [x] **Inter-rater reliability.**
  Per-dimension exact agreement rates reported for all 10 Layer 1 dimensions.

- [x] **Construct validity.**
  Inter-dimension Pearson correlations computed; one pair (CF × CO, r=0.75) disclosed and discussed.

---

## Code and Compute

- [x] **Code available with the paper.**
  Public GitHub repo (anonymized for review).

- [x] **Dependencies documented.**
  `requirements.txt` with pinned versions.

- [x] **Hardware specifications documented.**
  No local GPU required. Evaluation ran on a standard MacBook via Vultr API.

- [x] **Compute budget documented.**
  ~$200 on Vultr Serverless Inference for full evaluation.
  Wall-clock time: ~20 hours (parallelized across 2 process groups).

- [x] **Random seeds documented.**
  Bayesian posterior sampling: seed=42 (documented in `src/statistics.py`).
  Validation pilot stratified sampler: seed=42 (`validation_pilot/sample_pilot.py`).
  Fourth-judge stratified sampler: seed=4242 (`validation_pilot/sample_fourth_judge.py`).
  Model temperature 0.7 introduces stochasticity handled via 5 runs per prompt.

---

## Data

- [x] **Dataset publicly available.**
  Hosted on Hugging Face (with Croissant metadata).

- [x] **Dataset license clearly stated.**
  CC BY 4.0 (prompts, rubrics, results); MIT (code).

- [x] **Datasheet provided.**
  `submission/datasheet.md` following Gebru et al. 2021.

- [x] **Data collection process described.**
  Prompts authored by research team; model responses via API; judge scores via 3-model panel.

- [x] **Known biases and limitations described.**
  In datasheet Composition and Uses sections; in paper Discussion and Limitations.

---

## Results Reproducibility

- [x] **All raw data preserved.**
  Per-prompt responses, individual judge scores, and reasoning all preserved in result JSONs.

- [x] **Analysis scripts provided.**
  `run_analysis.py` generates all paper tables and CSV outputs.
  `validation_pilot/analyze.py` reproduces the agreement metrics in Appendix C.1 from the released pilot scores and fourth-judge outputs.

- [x] **Figure generation scripts provided.**
  `paper/generate_figures_v2.py` generates all 6 paper figures from analysis CSVs.

- [x] **Full prompt-response pairs available.**
  Each result JSON contains full prompt, response, score, reasoning, and evidence.

---

## Ethics and Broader Impact

- [x] **Ethics statement included** (`submission/ethics_statement.md`).

- [x] **Broader impact statement included** (`submission/broader_impact.md`).

- [x] **No human subjects.**
  This work evaluates AI outputs. The single-annotator validation pilot (Appendix C.1) is methodological self-validation by one of the paper's authors and does not constitute human subjects research; the planned multi-annotator study will undergo IRB review before execution.

- [x] **No PII.**
  All prompts use fictional scenarios.

- [x] **No offensive content.**
  Adversarial prompts are bounded professional scenarios; no slurs, threats, or explicit content.

---

## Paper-Specific

- [x] **Math notation defined.**
  PWI formula in Section 3 with all terms defined.

- [x] **Tables and figures have self-contained captions.**

- [x] **References to prior work appropriately credited.**
  40 references; central works (Bucinca et al., Acemoglu, Mollick, Sharma et al., Schemmer et al., Buijsman et al., Sturgeon et al.) cited multiple times with clear framing.
