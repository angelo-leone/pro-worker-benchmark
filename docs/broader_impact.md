# Broader Impact Statement

## Intended Impact

The Pro-Worker AI Benchmark aims to shift LLM evaluation toward human-centered measures of AI assistance quality. Current benchmarks (MMLU, HumanEval, MT-Bench) measure whether AI produces correct outputs; this benchmark asks whether AI keeps humans thinking, skilled, and in charge. By making "pro-worker" behavior measurable, we enable:

1. **More informed deployment decisions.** Organizations integrating LLMs into professional workflows can assess not just capability but interaction quality.
2. **Targeted model development.** Model trainers can identify specific behavioral gaps (e.g., cognitive forcing, complementarity) and develop interventions.
3. **Policy grounding.** Regulators and standards bodies can reference measurable criteria when discussing AI-worker interaction.

## Positive Impacts

**Preservation of human expertise.** If widely adopted, pro-worker-aligned LLMs could mitigate deskilling risks documented by Brynjolfsson et al. (2023) and Dell'Acqua et al. (2023). Workers using pro-worker AI may retain and grow skills rather than atrophy them.

**Democratization of expert mentorship.** Pro-worker AI behaviors (scaffolding, contrastive explanation, pattern-teaching) mirror what good human mentors provide. Access to such AI could extend mentorship-like benefits to workers who lack human mentors in their fields.

**Appropriate reliance calibration.** AI that actively helps users understand when to verify its outputs reduces the risk of over-trust errors that have caused significant harms in medical, legal, and financial contexts (Romeo et al. 2025).

**Reduced sycophancy.** Models trained or prompted toward anti-sycophancy produce more honest feedback, which is particularly valuable in educational contexts, performance reviews, and high-stakes decisions where flattery can cause real harm.

## Potential Negative Impacts

**Productivity trade-offs.** Pro-worker AI introduces friction (asking for user hypotheses, providing annotated drafts instead of finished products) that reduces short-term throughput. Organizations optimizing purely for speed may view this as a cost. The long-term benefits (skill retention, better decisions) may not be immediately visible.

**Paternalism risk.** Interventions like "cognitive forcing" could become paternalistic if applied inappropriately — e.g., forcing a doctor under time pressure to articulate hypotheses when they need rapid decision support. The system prompt we tested includes guidance on appropriate automation boundaries, but this balance is inherently difficult.

**Cultural assumptions.** The benchmark's prompts and rubrics reflect Anglo-American workplace norms (e.g., valuing individual skill development, Socratic questioning). Cultures that emphasize collective decision-making, deference to authority, or efficiency over pedagogy may view certain "pro-worker" behaviors differently.

**Benchmark gaming.** As with any benchmark, models could be trained to superficially exhibit pro-worker behaviors (asking token questions, adding disclaimers) without genuinely preserving human agency. Our judge rubric includes explicit distinctions between superficial and genuine behavior, but gaming remains a risk.

**Evaluation monoculture.** If this benchmark becomes dominant, it could narrow the space of what "good" AI behavior means. We encourage the research community to develop complementary benchmarks reflecting different values (e.g., pure productivity, safety, creative collaboration).

## Impact on Workers

This benchmark was designed with worker welfare in mind but does not directly involve workers in its creation. Important considerations:

- **Not a substitute for worker consultation.** Organizations deploying AI in workplaces should engage workers directly, not just consult benchmark scores.
- **May shift power dynamics.** Pro-worker AI preserves worker agency but does not address broader structural issues (wage compression, job displacement, surveillance).
- **Accessibility.** Pro-worker AI behaviors may benefit experienced workers more than novices. We specifically include a scaffolding dimension (skill preservation) to address this but acknowledge the gap may persist.

## Impact on Affected Communities

**Researchers in HCI and labor economics:** The benchmark provides a shared empirical tool for a research area previously dominated by qualitative studies and individual experiments.

**LLM developers:** The benchmark offers a new evaluation target. We note that system-prompt-based improvements (what this paper tests) are easier to adopt than retraining, making pro-worker alignment practically accessible.

**Downstream users:** Workers interacting with LLMs trained or prompted toward pro-worker behavior. These users are not directly consulted in benchmark creation — a limitation flagged in the ethics statement.

## Mitigations and Responsible Use

1. **Human validation priority.** We explicitly recommend pairing this benchmark with human expert validation before making deployment decisions (see `docs/human_validation_protocol.md`).

2. **Transparent limitations.** The paper's Discussion and Limitations sections explicitly address construct overlap, judge bias, English-only scope, and cultural assumptions.

3. **Open release.** All code, prompts, rubrics, and results are publicly available for scrutiny and extension.

4. **Versioning commitment.** We commit to updating the benchmark as LLMs evolve and as rubric refinements emerge from community use.

5. **Encourage contestation.** We explicitly invite researchers who disagree with the pro-worker framing to develop alternative benchmarks and evaluate against them comparatively.

## Climate / Environmental Impact

Full evaluation run: ~200 USD of API compute across 7 models × 2 conditions × 5 runs × 320 prompts × 3 judges. At approximate inference costs of ~0.1 kWh per $1 of LLM inference (rough industry estimate), this corresponds to ~20 kWh of compute — similar to one month of running a desktop computer. The benchmark's environmental footprint is negligible relative to the one-time training cost of any single foundation model evaluated.

## Commitment

We commit to:
- Maintaining the benchmark's public availability
- Addressing community-reported bugs and rubric ambiguities
- Publishing annual updates as LLM capabilities evolve
- Engaging with criticism transparently through the GitHub issue tracker
