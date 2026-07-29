# Benchmark Quality Standard: Pro-Worker AI Benchmark

This document codifies the acceptance criteria for a NeurIPS Datasets & Benchmarks-quality LLM evaluation suite. It draws on empirical findings from benchmark methodology research and serves as the quality gate for all benchmark components.

---

## 1. Construct Validity

A benchmark measures what it claims to measure. Each dimension must satisfy three forms of validity.

### 1.1 Content Validity
- Each dimension has a **falsifiable behavioral definition**: a clear description of what the AI must do (or not do) to score at each level.
- Definitions are grounded in peer-reviewed empirical findings, not intuition.
- Behavioral anchors at each score level (0-3) describe observable actions, not qualities (e.g., "asks for hypothesis before answering" not "is thoughtful").

### 1.2 Discriminant Validity
- Dimensions should measure **distinct constructs**. Inter-dimension Pearson correlations must be below **r < 0.70**.
- If two dimensions correlate above 0.70 across the full prompt set, they should be merged or one should be redesigned.
- Strongest risk pairs to monitor: `uncertainty_transparency` vs. `metacognitive_calibration`; `cognitive_forcing` vs. `complementarity`.

### 1.3 Convergent Validity (Internal Consistency)
- Prompts within a single dimension should correlate with each other. Target: **Cronbach's alpha > 0.70** per dimension.
- If alpha < 0.50, the prompts within that dimension are measuring different things and must be revised.
- Item analysis: each prompt's item-total correlation should be **> 0.30**. Prompts below this threshold are poor discriminators and should be replaced.

**Reference:** McIntosh et al. (2024), "Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence." 87 citations.

---

## 2. Scoring Reliability

Scores must be reproducible across judges and evaluation instances.

### 2.1 Multi-Judge Requirement
- Minimum **3 judge models** from different model families to avoid systematic bias.
- Report **Cohen's kappa** (pairwise) and **Krippendorff's alpha** (overall) as inter-rater reliability metrics.
- Acceptable thresholds: kappa > 0.60 (substantial agreement); alpha > 0.67 (tentative conclusions permitted), alpha > 0.80 (firm conclusions).
- If kappa < 0.60 for any dimension, the rubric for that dimension must be revised with clearer behavioral anchors and additional few-shot examples.

### 2.2 Score Distribution Analysis
- Report **score distributions** (histograms) per dimension per model. Flag:
  - **Ceiling effects**: > 80% of scores at maximum (3) — rubric too lenient or prompts too easy.
  - **Floor effects**: > 80% of scores at minimum (0) — rubric too strict or prompts misaligned.
  - **Bimodal distributions**: may indicate inconsistent judge behavior or ambiguous rubric boundaries.

### 2.3 Scoring Bias Mitigation
- **Rubric order randomization**: Present score levels in randomized order (ascending vs. descending) to mitigate order bias (Li et al. 2025).
- **Position bias testing**: Score a subset of responses with both rubric orders; flag cases where scores differ by > 1.
- **Length bias check**: Verify that score correlates with behavioral criteria, not response length. Compute Pearson r(score, word_count); flag if |r| > 0.30.
- **Self-evaluation bias**: Never use a model family as both judge and candidate. If unavoidable, report and analyze the bias.

### 2.4 Structured Output Enforcement
- Judge must return valid JSON with exactly `score`, `reasoning`, `evidence` fields.
- Implement retry logic for malformed responses (max 3 retries).
- Report parse failure rate per judge model; flag if > 5%.

**References:** 
- Gu et al. (2024), "A Survey on LLM-as-a-Judge." 717 citations.
- Li et al. (2025), "Evaluating Scoring Bias in LLM-as-a-Judge."
- Thakur et al. (2024), "Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges." 121 citations.

---

## 3. Statistical Methodology

Results must support valid statistical inference.

### 3.1 Multiple Runs
- Minimum **5 runs per prompt** to estimate within-prompt variance. Recommended: **10 runs** for publication-quality results.
- Report both individual run scores and aggregated statistics (mean, median, standard deviation).

### 3.2 Bayesian Confidence Intervals
- **Do not use CLT-based confidence intervals** for benchmarks with fewer than a few hundred data points per dimension. CLT dramatically underestimates uncertainty in small samples (Bowyer et al. 2025).
- Use **Dirichlet-Multinomial posterior** for ordinal score distributions (scores are categorical: 0, 1, 2, 3).
- Report **95% Highest Density Intervals (HDI)** for all dimension means and composite PWI scores.
- Use **bootstrap confidence intervals** (10,000 resamples) for composite PWI scores.

### 3.3 Effect Size Reporting
- For all comparisons (baseline vs. system-prompted, model A vs. model B), report **Cohen's d** effect size alongside p-values.
- Effect size interpretation: |d| < 0.20 negligible, 0.20-0.50 small, 0.50-0.80 medium, > 0.80 large.
- Report effect sizes per dimension AND for composite PWI.

### 3.4 Power Analysis
- Determine minimum prompts-per-dimension for detecting a meaningful difference (e.g., d = 0.50, alpha = 0.05, power = 0.80).
- With 20 prompts per dimension and 5 runs each (100 observations), a paired t-test achieves > 0.80 power for effects of d >= 0.28.

### 3.5 Multiple Comparisons
- When comparing multiple models, apply **Bonferroni correction** or **Benjamini-Hochberg FDR control** to p-values.
- Report both uncorrected and corrected significance levels.

**Reference:** Bowyer et al. (2025), "Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints." 12 citations.

---

## 4. Bias Mitigation & Contamination Control

### 4.1 Prompt Novelty
- Verify prompts do not appear verbatim in known training datasets or benchmark suites (MMLU, HellaSwag, MT-Bench, AlpacaEval).
- Compute n-gram overlap with known benchmarks; flag prompts with > 50% 5-gram overlap.
- All prompts should contain domain-specific details (company names, metrics, team sizes) that are fictional and unique to this benchmark.

### 4.2 Judge-Model Independence
- Judge models must come from **different model families** than candidate models.
- If a candidate model family must also serve as a judge (e.g., GPT-4o as candidate, GPT-4o-mini as judge), report this explicitly and analyze whether scores for that model family are systematically biased.

### 4.3 Prompt Memorization Testing
- For a subset of prompts, create "perturbed" variants (change domain, swap numbers, alter context) and verify that model responses differ substantively.
- If a model produces near-identical responses to semantically different prompts, this suggests memorization rather than genuine behavioral response.

### 4.4 Ordering Effects
- Randomize prompt presentation order across runs. Do not present prompts in the same order each time.
- For multi-turn scenarios, this is not applicable (order is inherent to the scenario).

---

## 5. Reporting Standards

### 5.1 Required Tables
1. **Model comparison table**: PWI scores with 95% CIs for all models x conditions.
2. **Per-dimension breakdown**: Mean scores (with CIs) per dimension per model.
3. **Effect size table**: Cohen's d for system-prompt impact per dimension per model.
4. **Inter-rater reliability table**: Kappa/alpha per dimension.
5. **Construct validity table**: Inter-dimension correlation matrix, Cronbach's alpha per dimension.

### 5.2 Required Figures
1. Radar/spider chart of dimension scores per model.
2. Score distribution histograms per dimension (aggregate across models).
3. System prompt delta heatmap (dimensions x models).
4. Inter-rater reliability visualization.

### 5.3 Reproducibility
- All prompts, rubrics, scoring code, and evaluation results released as open-source.
- Exact model IDs, API parameters (temperature, max_tokens), and random seeds documented.
- Results files include timestamps, model versions, and judge configurations.

---

## 6. Checklist

Before submission, verify:

- [ ] All 11 dimensions have falsifiable behavioral definitions with 0-3 anchors
- [ ] All 11 dimensions have 4 few-shot calibration examples (one per score level)
- [ ] Minimum 20 prompts per Layer 1 dimension
- [ ] All prompts include difficulty, domain, and primary_dimension metadata
- [ ] No prompt is context-free (all provide sufficient info for substantive response)
- [ ] 3+ judge models from different families
- [ ] Cohen's kappa > 0.60 for all judge pairs on all dimensions
- [ ] Minimum 5 runs per prompt
- [ ] Bayesian confidence intervals reported (not CLT-based)
- [ ] Effect sizes (Cohen's d) reported for all comparisons
- [ ] Inter-dimension correlations < 0.70 (discriminant validity)
- [ ] Cronbach's alpha > 0.70 per dimension (convergent validity)
- [ ] Score-length correlation |r| < 0.30 for all dimensions
- [ ] Prompt novelty verified against known benchmarks
- [ ] Judge-model family independence maintained
- [ ] Weight sensitivity analysis with 3+ alternative schemes
- [ ] All results reproducible from released code and prompts
