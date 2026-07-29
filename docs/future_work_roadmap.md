# Pro-Worker AI Benchmark — Future Work Roadmap

The full prose extracted from §7 of the paper before it was compressed for page-limit reasons. Kept as a planning reference; not the canonical version (the paper has the abridged form).

---

## 1. Human validation study

Recruit domain experts to score a subset of responses alongside the LLM judges, establishing inter-rater reliability between human and automated evaluation and identifying systematic judge biases. The released datasheet describes a recommended pilot of 2–3 annotators × 30 responses each as a low-cost first step (~10–15 hours of expert time).

## 2. Field user study

Deploy the pro-worker system prompt in real workplace settings and measure downstream outcomes: decision quality, skill retention, user satisfaction, and long-term engagement patterns. The benchmark establishes that prompted models *exhibit* augmentative behavior; the field study would establish whether that behavior *causes* the intended downstream effects (the predictive-validity test).

## 3. Rubric refinement for low-IRR dimensions

Develop more granular rubrics for appropriate reliance (18.7% IRR) and uncertainty transparency (15.0% IRR), decomposing these holistic judgments into sets of binary sub-criteria that achieve higher inter-rater agreement. Likely path: produce 3–5 binary sub-criteria per dimension, each with a yes/no anchor and a calibration example, then validate that the combined score correlates >0.7 with the original 0–3 holistic judgment while improving exact-agreement IRR above 50%.

## 4. Proprietary model evaluation

Extend evaluation to proprietary models (GPT-4o, Claude 3.5 Sonnet, Gemini Pro) to assess whether different training approaches yield systematically different pro-worker profiles. Hypothesis: Anthropic-trained models score higher on uncertainty transparency and ethical surfacing because of constitutional AI; OpenAI-trained models score higher on cognitive forcing under prompted conditions because of stronger instruction following; Google-trained models score in between.

## 5. Multilingual extension

Adapt the benchmark for languages and cultural contexts where norms around authority, deference, and directness differ from the English-language Western context. Highest-priority targets: Japanese (high-deference workplace culture, indirect communication norms), Mandarin (managerial authority dynamics), Spanish (Latin American business norms). Each adaptation requires native-speaker prompt review, not just translation.

## 6. Domain-specific benchmarks

Create specialized versions for high-stakes domains where the consequences of substitution are severe: clinical decision support (medication dosing, differential diagnosis), legal analysis (case research, contract review), and financial advising (investment recommendations, risk assessment). Each variant would extend the rubric with domain-specific calibration anchors and add ~50 dimension-relevant prompts.

## 7. Longitudinal deskilling studies

Measure whether models scoring high on the PWI actually prevent deskilling in controlled longitudinal experiments. Recommended design: pair-randomized, three-arm RCT (no AI / low-PWI AI / high-PWI AI) with 6-month skill retention measurement on out-of-distribution tasks. This is the predictive-validity test that would convert PWI from a behavioral benchmark into an outcome-validated one.

## 8. Adaptive PWI

Incorporate Buçinca et al.'s (2024) reinforcement-learning approach to create AI systems that dynamically adjust their pro-worker engagement level based on inferred user state. The current PWI is a static score across a fixed prompt set; adaptive PWI would measure how well a model varies its engagement strategy across user contexts (skill level, motivation, time pressure) and reward smart adaptation rather than uniform behavior.

---

## Compressed paragraph form (what now appears in the paper)

> Several extensions are planned: a human validation study comparing expert and LLM-judge scores; a field deployment measuring downstream skill retention; rubric refinement for the two low-IRR dimensions; proprietary-model evaluation (GPT-4o, Claude 3.5 Sonnet, Gemini); multilingual adaptation for non-Western workplace contexts; domain-specific variants for clinical, legal, and financial applications; longitudinal deskilling studies establishing predictive validity; and an adaptive PWI building on Buçinca et al.'s reinforcement-learning approach to vary engagement by user state.
