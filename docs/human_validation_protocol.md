# Human Validation Protocol

This document specifies the protocol for validating the LLM-as-judge scoring
against human expert judgment. It is required for NeurIPS Datasets & Benchmarks
submission to establish that automated scoring aligns with human assessment.

---

## 1. Objective

Quantify the agreement between LLM judge scores and human expert scores across
all 11 dimensions, establishing:
- **Human-human reliability** (baseline: how much do human experts agree?)
- **Human-LLM reliability** (validation: does the LLM judge agree with humans?)
- **Systematic biases** (does the LLM judge over/under-score specific dimensions?)

---

## 2. Sample Selection

### 2.1 Response Sample
- **20 prompts** selected from Layer 1 (stratified: 2 per behavioral dimension,
  ensuring coverage of easy/medium/hard difficulty levels)
- **3 models** (1 open-weight, 1 proprietary, 1 small — to capture score range)
- **2 conditions** (baseline, with system prompt)
- **Total: 120 response pairs** to evaluate

### 2.2 Selection Criteria
- Include prompts from at least 5 different professional domains
- Include at least 2 prompts from each new dimension (anti_sycophancy,
  metacognitive_calibration, appropriate_reliance, ethical_surfacing)
- Include responses spanning the full score range (0-3) based on LLM judge scores

---

## 3. Annotators

### 3.1 Composition
- **3 domain experts** per professional domain represented in the prompts:
  - Business/strategy (MBA or 5+ years management consulting)
  - Software engineering (5+ years professional development)
  - Data science (MS/PhD in quantitative field or 3+ years applied ML)
  - Healthcare (clinical or health informatics background)
  - Legal (JD or 5+ years legal practice)

- **2 HCI researchers** with expertise in:
  - Human-AI interaction
  - Intelligent tutoring systems or adaptive learning
  - Evaluation methodology

### 3.2 Training Protocol
1. Annotators read the benchmark quality standard document
2. Annotators review all 11 dimension rubrics with behavioral anchors
3. Annotators score 5 calibration examples (with gold-standard scores) and
   discuss disagreements
4. Inter-rater reliability on calibration set must reach kappa > 0.60 before
   proceeding to the main annotation task

---

## 4. Annotation Task

### 4.1 Materials Per Response
Each annotator receives:
- The user prompt (full text)
- The AI response (full text)
- The dimension being evaluated
- The scoring rubric for that dimension (0-3 with behavioral anchors)
- NO access to the LLM judge's score (blind evaluation)

### 4.2 Scoring
- Each annotator independently assigns a score (0-3) per the rubric
- Each annotator provides a 1-2 sentence justification
- Each annotator flags any rubric ambiguity encountered

### 4.3 Coverage
- Each response is scored by at least 3 annotators
- Domain-specific prompts are scored by at least 1 domain expert
- All annotators score all dimensions (not just their domain)

---

## 5. Analysis

### 5.1 Human-Human Agreement
- **Pairwise Cohen's kappa** for all annotator pairs per dimension
- **Krippendorff's alpha** (ordinal) overall and per dimension
- **Exact agreement rate** and **adjacent agreement rate** (within ±1)
- Target: kappa > 0.60 (substantial agreement)

### 5.2 Human-LLM Agreement
- Compare LLM judge's score to **human majority vote** per response
- **Cohen's kappa** between LLM judge and human consensus
- **Mean absolute difference** between LLM and human scores
- **Systematic bias**: mean(LLM score - human score) per dimension
  - Positive bias = LLM judges more leniently than humans
  - Negative bias = LLM judges more strictly than humans

### 5.3 Dimension-Level Analysis
- Which dimensions have highest/lowest human-LLM agreement?
- Hypothesis: newer dimensions (anti_sycophancy, metacognitive_calibration)
  may have lower agreement due to less established evaluation norms
- Action: revise rubrics for dimensions with kappa < 0.50

### 5.4 Reporting
Results table format:

| Dimension | Human-Human κ | Human-LLM κ | Mean Bias | N |
|-----------|---------------|-------------|-----------|---|
| cognitive_forcing | 0.XX | 0.XX | +/-0.XX | 12 |
| ... | ... | ... | ... | ... |

---

## 6. Quality Controls

- **Random ordering**: Responses presented in random order (not grouped by model
  or dimension)
- **No model identification**: Annotators do not know which model produced which
  response
- **No condition identification**: Annotators do not know if system prompt was used
- **Attention checks**: 3 calibration responses with known gold-standard scores
  embedded randomly in the annotation set
- **Time tracking**: Record time per annotation to identify rushed responses

---

## 7. Iteration Protocol

If human-LLM kappa < 0.50 for any dimension:
1. Analyze disagreement patterns (where does the LLM differ from humans?)
2. Revise the rubric behavioral anchors for that dimension
3. Add/replace few-shot calibration examples
4. Re-run LLM judge on the same responses with revised rubric
5. Re-compute agreement metrics
6. If still < 0.50 after revision, report as a limitation in the paper

---

## 8. Timeline and Cost

- **Recruitment**: 2 weeks (academic collaborators, Prolific for domain experts)
- **Training**: 1 session (2 hours, including calibration)
- **Annotation**: ~3 hours per annotator (120 responses × ~90 seconds each)
- **Analysis**: 1 week
- **Estimated cost**: $50-75/hour × 5 annotators × 5 hours = $1,250-1,875
  (excluding HCI researchers who may be co-authors)

---

## 9. IRB Considerations

- This study evaluates AI system outputs, not human subjects
- Annotators are compensated professionals performing expert evaluation
- No personally identifiable information is collected beyond annotator IDs
- Check institutional requirements — some universities require IRB review
  for any human data collection, including expert annotation tasks
