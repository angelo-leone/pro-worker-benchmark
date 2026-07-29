# Datasheet for the Pro-Worker AI Benchmark

Following the datasheet template of Gebru et al. (2021), "Datasheets for Datasets."

---

## Motivation

**For what purpose was the dataset created?**
The Pro-Worker AI Benchmark was created to evaluate whether large language models augment human cognitive abilities or substitute for them. Existing LLM benchmarks measure task performance (accuracy, fluency) but not whether AI interaction patterns support or undermine human cognitive engagement, skill retention, and agency. This benchmark fills that gap by operationalizing findings from HCI and labor economics research into a systematic, reproducible evaluation framework.

**Who created the dataset and on behalf of which entity?**
[Anonymous for submission]

**Who funded the creation of the dataset?**
[Anonymous for submission] — Budget from author's personal cloud credits (~$200 USD total for evaluation compute).

---

## Composition

**What do the instances that comprise the dataset represent?**
The dataset comprises three types of instances:
1. **Prompts** (320 total): User queries across 11 behavioral dimensions, designed to probe pro-worker behaviors.
2. **Rubrics**: 11 scoring rubrics with 0-3 behavioral anchors and calibration examples.
3. **Model responses + judge scores**: ~96,000 scored (prompt, response, score, reasoning) tuples from 7 LLMs across 2 conditions.

**How many instances are there in total?**
- Layer 1 (single-turn behavioral probes): 200 prompts × 10 dimensions
- Layer 2 (multi-turn scenarios): 16 scenarios × 5 turns = 80 turns
- Layer 3 (adversarial stress tests): 40 prompts
- Total evaluation instances: 320
- Total scored data points: ~96,000 (7 models × 2 conditions × 320 × 5 runs × ~3 judges)

**Does the dataset contain all possible instances or is it a sample?**
The dataset is a curated sample designed to cover:
- 11 behavioral dimensions grounded in peer-reviewed literature
- 13 professional domains (business, engineering, data science, medical, legal, etc.)
- 3 difficulty tiers (easy, medium, hard)
- Multiple user archetypes (novice, developing, proficient, expert)

**What data does each instance consist of?**
Each prompt instance includes:
- `id`: unique identifier
- `prompt`: the user query text
- `domain`: professional domain
- `difficulty`: easy/medium/hard
- `primary_dimension`: which behavioral dimension is being tested
- `context`: annotator notes on ideal pro-worker behavior

Each scored response instance includes:
- Full AI response text
- Judge score (0-3) with reasoning and evidence
- Individual scores from 3 judges
- Inter-rater agreement metrics

**Is there a label or target associated with each instance?**
Each response is scored on a 0-3 ordinal scale by 3 independent LLM judges, with median aggregation. Scores are accompanied by judge reasoning and quoted evidence.

**Is any information missing from individual instances?**
Approximately 0.3-0.7% of judge calls returned parse failures (scored -1 and excluded from analysis). Documented in the paper.

**Are relationships between individual instances made explicit?**
Yes. Layer 2 scenarios have explicit turn sequencing. Layer 1 prompts are grouped by dimension.

**Are there recommended data splits?**
The benchmark is a fixed evaluation suite — no train/val/test splits. All instances are held out by design.

**Are there any errors, sources of noise, or redundancies?**
- Approximately 1% of judge calls fail to return parseable JSON; these are retried up to 3 times
- Two dimensions (appropriate_reliance, uncertainty_transparency) show lower inter-rater reliability (<20% exact agreement) and are flagged as rubric-refinement priorities
- One dimension pair (cognitive_forcing × complementarity) shows r=0.75 correlation, indicating partial construct overlap documented in the paper

**Is the dataset self-contained, or does it link to other external resources?**
Self-contained. The prompts, rubrics, calibration examples, and evaluation code are all included in the release. Model responses were obtained via API calls to Vultr Serverless Inference; those models are open-weight and accessible via multiple providers.

**Does the dataset contain data that might be considered confidential?**
No. All prompts are fictional scenarios. No real person, company, or private information.

**Does the dataset contain data that might be offensive, insulting, threatening, or cause anxiety?**
The Layer 3 adversarial prompts include scenarios designed to stress-test AI responses to emotional pressure, authority claims, and demanding users. These are bounded professional scenarios (e.g., stressed manager, impatient executive). No slurs, no physical threats, no sexually explicit content.

---

## Collection Process

**How was the data associated with each instance acquired?**
- **Prompts**: Authored by the research team based on realistic professional scenarios. Cross-reviewed for realism and dimension alignment.
- **Rubrics**: Developed iteratively from the research literature (Buçinca et al. 2021/2024, Acemoglu 2024, Sharma et al. 2023, Schemmer et al. 2023, Buijsman et al. 2025, Sturgeon et al. 2025).
- **Model responses**: Generated by 7 LLMs via Vultr Serverless Inference API with temperature 0.7, max_tokens 8192.
- **Judge scores**: Generated by 3 LLMs (Devstral-2 123B, GPT-oss 120B, Gemma 4 31B) with temperature 0.0.

**What mechanisms were used to collect the data?**
Automated Python pipeline (`src/runner.py`) with:
- Parallel execution (2 process groups)
- Multi-judge panel scoring (median aggregation)
- Automatic retry on transient API failures
- Incremental checkpointing after each layer
- 99%+ judge success rate achieved

**Over what timeframe was the data collected?**
April 16-19, 2026 (final evaluation run, v4).

**Validation pilot.** A single-annotator pilot (N=30, one of the paper's authors, blind to LLM-judge outputs) and a Claude-as-fourth-judge cross-family consistency check (N=500) were added on April 28, 2026, packaged in `validation_pilot/` of the supplementary release. Stratified samples are reproducible from the released result files using `validation_pilot/sample_pilot.py` (seed=42) and `validation_pilot/sample_fourth_judge.py` (seed=4242); analyses are reproduced via `validation_pilot/analyze.py`. Reported metrics: human pilot exact 56.7\%, adjacent 86.7\%, quadratic-weighted κ=0.702; fourth-judge exact 66.8\%, adjacent 96.8\%, κ=0.840. Full protocol and per-dimension breakdown in Appendix C.1 of the paper.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing or cleaning of the data done?**
- Judge responses parsed for JSON extraction with fallback regex
- Failed judge calls (score = -1) excluded from aggregated statistics
- Outlier model (Qwen3.5 397B) excluded due to 55% judge failure rate documented transparently

**Was the "raw" data saved in addition to the preprocessed/cleaned data?**
Yes. All raw model responses, individual judge scores, and reasoning are preserved in the JSON result files.

---

## Uses

**Has the dataset been used for any tasks already?**
Yes — the present paper, which reports the full evaluation of 7 LLMs.

**Is there a repository that links to all uses?**
The benchmark is openly released for community use. Future uses will be tracked via citations.

**What (other) tasks could the dataset be used for?**
- Evaluating new LLMs released after the benchmark's publication
- Testing prompt-engineering techniques against a standardized rubric
- Training pro-worker-aligned models via RLHF using the rubrics as reward signal
- Human validation studies comparing LLM judge scores to human expert judgment

**Is there anything about the composition or collection that might impact future uses?**
- English-only prompts — cultural assumptions about directness/deference may not transfer
- Judges and candidate models partially overlap (GPT-oss and Gemma serve both roles) — documented and analyzed as self-evaluation bias in the paper
- Benchmarks degrade as models improve; periodic re-validation recommended

**Are there tasks for which the dataset should not be used?**
- NOT for training proprietary models without credit
- NOT as a sole decision-making input for model deployment (complement with human validation)
- NOT for evaluating models in languages other than English (out of scope)

---

## Distribution

**Will the dataset be distributed to third parties outside the entity?**
Yes, open release under CC BY 4.0 (code under MIT).

**How will the dataset be distributed?**
- Hugging Face Hub: [URL to be added post-acceptance]
- GitHub: [URL to be added post-acceptance, anonymous during review]
- Zenodo: DOI for archival version

**When will the dataset be distributed?**
Publicly available at paper submission (with anonymized hosting for double-blind review). Finalized at camera-ready.

**Will the dataset be distributed under a copyright or other intellectual property license?**
- Prompts, rubrics, results: CC BY 4.0
- Code: MIT License

**Have any third parties imposed IP-based or other restrictions on the data?**
No.

---

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The original authors. Hugging Face Hub provides long-term hosting.

**How can the owner/curator/manager be contacted?**
Via the GitHub repository issue tracker (post-acceptance) or the corresponding author's email (post-camera-ready).

**Will the dataset be updated?**
- v2.0 (current): 11 dimensions, 7 models, multi-judge panel, single-annotator validation pilot, Claude-as-fourth-judge cross-family check
- v3.0 (planned): Multi-annotator human validation study, refined rubrics for low-IRR dimensions (UT, AR\_d, MC)
- Future versions: as new models/LLM capabilities emerge

**Will older versions continue to be supported?**
Yes, versioned releases on Hugging Face. v1.0 (February 2026) remains available for historical reference but is superseded by v2.0.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism to do so?**
Yes — GitHub pull requests for prompt additions, rubric refinements, or new dimensions. Contributions subject to maintainer review.
