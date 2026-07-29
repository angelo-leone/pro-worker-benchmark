# Ethics Statement

## Human Subjects Research

This work does not involve human subjects in the regulatory sense. The evaluation measures AI model outputs against predefined rubrics. The single-annotator validation pilot reported in Appendix C.1 (N=30, scored by one of the paper's authors) is methodological self-validation against already-collected model outputs and does not constitute human subjects research under 45 CFR 46. The full multi-annotator study described in Section 7 (Future Work) will involve external expert annotators and will undergo Institutional Review Board (IRB) review before execution.

## Data Privacy

All prompts in the benchmark are authored by the research team and feature fictional scenarios with invented companies, metrics, and personas. No prompt contains personally identifiable information (PII), no real-world private health data, no real customer information, and no real employee records. Prompts that resemble workplace situations (e.g., hiring decisions, performance reviews) describe invented cases.

## Potentially Sensitive Content

The Layer 3 adversarial stress tests include prompts designed to test AI resilience under pressure — simulated urgency, authority claims, emotional appeals, and demanding user behavior. These prompts describe bounded professional scenarios (stressed managers, impatient executives) and do not contain:

- Slurs or hate speech
- Threats of physical violence
- Sexually explicit material
- Content targeting specific real individuals

However, some prompts describe scenarios (e.g., data breach communications, hiring disputes, medical triage decisions) that could be triggering for readers with relevant personal experiences. Readers are advised to exercise discretion when reviewing the Layer 3 prompts in `prompts/layer3_adversarial/stress_tests.yaml`.

## Value Commitments

This benchmark operationalizes a specific theoretical position: that AI should augment rather than substitute for human cognition, preserve skill development, and maintain human agency over consequential decisions. This framing draws from HCI research (Buçinca et al.), labor economics (Acemoglu, Mollick), and pro-worker AI policy frameworks.

We acknowledge that reasonable researchers may disagree with aspects of this framing. For example:
- Some argue that full AI automation of cognitive tasks is a desirable productivity goal
- Some contexts (emergencies, routine tasks) may legitimately call for substitutional rather than augmentative AI
- The "pro-worker" framing reflects Anglo-American workplace norms and may not transfer to all cultural contexts

The benchmark is not intended as a universal arbiter of "good" AI behavior. Rather, it offers a measurable, reproducible tool for evaluating one coherent value position about AI alignment with human cognitive partnership.

## Dual Use Considerations

**Potential positive uses:**
- Evaluating AI systems for deployment in professional workflows
- Identifying training gaps in current RLHF pipelines
- Informing policy discussions on AI-worker interaction

**Potential misuse:**
- Benchmark gaming via prompt overfitting — we encourage pairing with held-out human evaluation
- Selective use of benchmark results without reporting the full scope of measurements

We release the benchmark openly to enable both scrutiny and community extension, rather than restricting access to prevent hypothetical misuse.

## Judge-Candidate Overlap

Two of our three judge models (GPT-oss 120B, Gemma 4 31B) are also among the candidate models evaluated. This creates potential self-evaluation bias. We document this explicitly in the paper and recommend interpreting those specific models' scores with awareness of this confound. The paper discusses this as a limitation and points to future work with fully independent judge panels.

## Environmental Impact

The evaluation consumed approximately 200 USD of API compute, corresponding to a modest carbon footprint relative to training a foundation model. All model calls were made to models already trained by their respective creators; this benchmark adds only inference-time compute.

## Model Provenance

All seven evaluated models are open-weight, meaning:
- Their weights are publicly available
- Their training data (where disclosed) can be scrutinized
- Replication of this work does not require access to proprietary APIs

We did not evaluate proprietary models (GPT-4o, Claude, Gemini) due to access constraints. This limitation is documented in the paper and flagged as a priority for future work.

## Institutional Context

[Anonymous for submission.]
