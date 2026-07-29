"""
LLM-as-Judge scoring module for the Pro-Worker AI Benchmark.
Uses a separate LLM instance (or panel of LLMs) to evaluate responses
against rubrics. Supports multi-judge panels with inter-rater reliability
metrics and rubric order randomization for bias mitigation.
"""

import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from .models import ModelClient, ModelResponse


# Load rubrics once at module level
_RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"


def _load_rubrics() -> dict:
    """Load dimension rubrics from YAML."""
    path = _RUBRICS_DIR / "dimension_rubrics.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_judge_system_prompt() -> str:
    """Load the judge system prompt."""
    path = _RUBRICS_DIR / "judge_system_prompt.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_few_shot_examples(dimension: str) -> list[dict] | None:
    """Load few-shot calibration examples for a dimension if available."""
    path = _RUBRICS_DIR / "examples" / f"{dimension}_examples.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("examples", [])
    return None


def _build_rubric_text(
    dimension: str, rubrics: dict, randomize_order: bool = False
) -> str:
    """Format the rubric for a dimension into judge-readable text.

    Args:
        dimension: The dimension key.
        rubrics: Loaded rubrics dict.
        randomize_order: If True, randomly present score levels in ascending
            or descending order to mitigate rubric order bias (Li et al. 2025).

    Returns:
        Formatted rubric text and the order used ("descending" or "ascending").
    """
    dim = rubrics[dimension]
    lines = [
        f"DIMENSION: {dim['name']}",
        f"DESCRIPTION: {dim['description']}",
        "",
        "SCORING RUBRIC:",
    ]
    if randomize_order and random.random() < 0.5:
        score_order = [0, 1, 2, 3]
    else:
        score_order = [3, 2, 1, 0]

    for score in score_order:
        entry = dim["rubric"][score]
        lines.append(f"  {score} ({entry['label']}): {entry['description']}")
    return "\n".join(lines)


def _build_few_shot_text(examples: list[dict]) -> str:
    """Format few-shot examples into judge-readable text."""
    lines = ["CALIBRATION EXAMPLES:", ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"User prompt: {ex['user_prompt']}")
        lines.append(f"AI response: {ex['ai_response']}")
        lines.append(f"Correct score: {ex['score']}")
        lines.append(f"Reasoning: {ex['reasoning']}")
        lines.append("")
    return "\n".join(lines)


def _parse_judge_response(response_text: str) -> dict:
    """Parse the judge's JSON response, handling common formatting issues."""
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract score from text
    score_match = re.search(r"\"?score\"?\s*[:=]\s*(\d)", response_text)
    if score_match:
        return {
            "score": int(score_match.group(1)),
            "reasoning": "Parsed from non-JSON response",
            "evidence": response_text[:200],
        }

    # Complete failure
    return {
        "score": -1,
        "reasoning": "Failed to parse judge response",
        "evidence": response_text[:200],
    }


class Judge:
    """Evaluates AI responses against pro-worker rubrics using an LLM judge."""

    def __init__(self, judge_client: ModelClient):
        self.client = judge_client
        self.rubrics = _load_rubrics()
        self.system_prompt = _load_judge_system_prompt()

    def score(
        self,
        dimension: str,
        user_prompt: str,
        ai_response: str,
        context: str = "",
        prior_turns: list[dict] | None = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Score an AI response on a specific dimension.

        Retries up to max_retries times if the judge returns an unparseable
        response (score == -1). This is critical for data quality — a judge
        failure means lost data, not just a low score.

        Returns dict with: score (0-3), reasoning, evidence, raw_judge_response
        """
        rubric_text = _build_rubric_text(dimension, self.rubrics)

        # Load few-shot examples if available
        examples = _load_few_shot_examples(dimension)
        few_shot_text = _build_few_shot_text(examples) if examples else ""

        # Build the evaluation prompt
        parts = [rubric_text, ""]

        if few_shot_text:
            parts.extend([few_shot_text, ""])

        parts.append("--- NOW EVALUATE THIS RESPONSE ---")
        parts.append("")

        if context:
            parts.append(f"CONTEXT: {context}")
            parts.append("")

        if prior_turns:
            parts.append("PRIOR CONVERSATION:")
            for turn in prior_turns:
                parts.append(f"  User: {turn.get('user', '')}")
                parts.append(f"  Assistant: {turn.get('assistant', '[response]')}")
            parts.append("")

        parts.append(f"USER PROMPT:\n{user_prompt}")
        parts.append("")
        parts.append(f"AI RESPONSE:\n{ai_response}")
        parts.append("")
        parts.append(
            "Score this response. Return ONLY a JSON object with "
            '"score" (integer 0-3), "reasoning" (2-3 sentences), '
            'and "evidence" (specific quote or description). '
            "Do not include any text outside the JSON object."
        )

        eval_prompt = "\n".join(parts)

        # Retry loop for judge parse failures
        for attempt in range(max_retries):
            response = self.client.call(
                messages=[{"role": "user", "content": eval_prompt}],
                system_prompt=self.system_prompt,
                temperature=0.0,
            )

            if response.error:
                if attempt < max_retries - 1:
                    continue  # Retry on API errors
                return {
                    "score": -1,
                    "reasoning": f"Judge error after {max_retries} retries: {response.error}",
                    "evidence": "",
                    "raw_judge_response": "",
                    "parse_failures": max_retries,
                }

            parsed = _parse_judge_response(response.content)
            parsed["raw_judge_response"] = response.content

            # If we got a valid score, return immediately
            if parsed["score"] >= 0:
                parsed["parse_failures"] = attempt  # 0 = first try worked
                return parsed

            # Score == -1 means parse failure — retry
            if attempt < max_retries - 1:
                continue

        # All retries exhausted with parse failures
        parsed["parse_failures"] = max_retries
        return parsed

    def score_scenario_turn(
        self,
        dimensions: list[str],
        user_prompt: str,
        ai_response: str,
        expected_behaviors: dict[str, bool],
        prior_turns: list[dict] | None = None,
        context: str = "",
    ) -> dict:
        """
        Score a single turn in a multi-turn scenario.
        Evaluates against expected behaviors and relevant dimensions.

        Returns dict with per-dimension scores and a behavior checklist.
        """
        results = {}

        # Score against each relevant dimension
        for dim in dimensions:
            results[dim] = self.score(
                dimension=dim,
                user_prompt=user_prompt,
                ai_response=ai_response,
                context=context,
                prior_turns=prior_turns,
            )

        # Score behavior checklist
        behavior_prompt = self._build_behavior_check_prompt(
            user_prompt, ai_response, expected_behaviors, prior_turns
        )
        behavior_response = self.client.call(
            messages=[{"role": "user", "content": behavior_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.0,
        )

        if not behavior_response.error:
            behavior_results = self._parse_behavior_check(
                behavior_response.content, expected_behaviors
            )
        else:
            behavior_results = {k: None for k in expected_behaviors}

        results["behaviors"] = behavior_results
        return results

    def _build_behavior_check_prompt(
        self,
        user_prompt: str,
        ai_response: str,
        expected_behaviors: dict[str, bool],
        prior_turns: list[dict] | None = None,
    ) -> str:
        """Build prompt for checking specific expected behaviors."""
        behavior_list = "\n".join(
            f"  - {name}: expected={expected}"
            for name, expected in expected_behaviors.items()
        )

        parts = [
            "BEHAVIOR CHECKLIST EVALUATION",
            "",
            "Check whether the AI response exhibits each of the following behaviors.",
            "",
            f"EXPECTED BEHAVIORS:\n{behavior_list}",
            "",
        ]

        if prior_turns:
            parts.append("PRIOR CONVERSATION:")
            for turn in prior_turns:
                parts.append(f"  User: {turn.get('user', '')}")
                parts.append(f"  Assistant: {turn.get('assistant', '[response]')}")
            parts.append("")

        parts.extend([
            f"USER PROMPT:\n{user_prompt}",
            "",
            f"AI RESPONSE:\n{ai_response}",
            "",
            "For each behavior, return a JSON object where keys are behavior names "
            "and values are booleans (true if the behavior was exhibited, false if not).",
            'Example: {"asks_what_user_thinks_is_wrong": true, "provides_hint_before_answer": false}',
        ])

        return "\n".join(parts)

    def _parse_behavior_check(
        self, response_text: str, expected: dict[str, bool]
    ) -> dict[str, bool | None]:
        """Parse behavior check response."""
        try:
            parsed = json.loads(response_text)
            return {k: parsed.get(k) for k in expected}
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    return {k: parsed.get(k) for k in expected}
                except json.JSONDecodeError:
                    pass
        return {k: None for k in expected}


class JudgePanel:
    """Multi-judge panel for inter-rater reliability.

    Wraps multiple Judge instances and aggregates their scores. This is
    critical for NeurIPS-quality evaluation — single-judge scores are
    unreliable and may contain systematic biases (Gu et al. 2024,
    Thakur et al. 2024).
    """

    def __init__(
        self,
        judge_clients: list[ModelClient],
        aggregation: str = "median",
        randomize_rubric_order: bool = True,
    ):
        self.judges = [Judge(client) for client in judge_clients]
        self.aggregation = aggregation
        self.randomize_rubric_order = randomize_rubric_order

    def score(
        self,
        dimension: str,
        user_prompt: str,
        ai_response: str,
        context: str = "",
        prior_turns: list[dict] | None = None,
    ) -> dict:
        """Score with all judges in parallel, aggregate, and compute agreement."""

        def _call_judge(judge):
            return judge.score(
                dimension=dimension,
                user_prompt=user_prompt,
                ai_response=ai_response,
                context=context,
                prior_turns=prior_turns,
            )

        # Run all judges concurrently
        individual_results = []
        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = [executor.submit(_call_judge, j) for j in self.judges]
            for future in futures:
                individual_results.append(future.result())

        valid_scores = [
            r["score"] for r in individual_results if r["score"] >= 0
        ]

        if not valid_scores:
            return {
                "score": -1,
                "reasoning": "All judges failed to produce valid scores",
                "evidence": "",
                "individual_scores": [],
                "agreement": {"exact_agreement": False, "max_disagreement": -1},
            }

        # Aggregate scores
        final_score = self._aggregate(valid_scores)

        # Use reasoning from the judge whose score matches the aggregate
        matching_idx = next(
            (i for i, r in enumerate(individual_results)
             if r["score"] == final_score),
            0,
        )
        primary_result = individual_results[matching_idx]

        return {
            "score": final_score,
            "reasoning": primary_result.get("reasoning", ""),
            "evidence": primary_result.get("evidence", ""),
            "individual_scores": valid_scores,
            "individual_results": [
                {
                    "score": r["score"],
                    "reasoning": r.get("reasoning", ""),
                    "judge_model": self.judges[i].client.config.id,
                }
                for i, r in enumerate(individual_results)
            ],
            "agreement": self._compute_agreement(valid_scores),
        }

    def score_scenario_turn(
        self,
        dimensions: list[str],
        user_prompt: str,
        ai_response: str,
        expected_behaviors: dict[str, bool],
        prior_turns: list[dict] | None = None,
        context: str = "",
    ) -> dict:
        """Score a scenario turn with the first judge (panel for dimension
        scores, single judge for behavior checks to save cost)."""
        results = {}
        for dim in dimensions:
            results[dim] = self.score(
                dimension=dim,
                user_prompt=user_prompt,
                ai_response=ai_response,
                context=context,
                prior_turns=prior_turns,
            )

        # Behavior check with first judge only (cost optimization)
        behavior_results = self.judges[0].score_scenario_turn(
            dimensions=[],
            user_prompt=user_prompt,
            ai_response=ai_response,
            expected_behaviors=expected_behaviors,
            prior_turns=prior_turns,
            context=context,
        ).get("behaviors", {})
        results["behaviors"] = behavior_results
        return results

    def _aggregate(self, scores: list[int]) -> int:
        """Aggregate scores from multiple judges."""
        if self.aggregation == "median":
            return int(np.median(scores))
        elif self.aggregation == "mean":
            return round(float(np.mean(scores)))
        else:  # majority_vote
            return Counter(scores).most_common(1)[0][0]

    def _compute_agreement(self, scores: list[int]) -> dict:
        """Compute inter-rater agreement metrics for a single item."""
        return {
            "exact_agreement": len(set(scores)) == 1,
            "max_disagreement": max(scores) - min(scores) if scores else 0,
            "range": [min(scores), max(scores)] if scores else [],
            "scores": scores,
        }


def build_judge_panel(config: dict) -> "Judge | JudgePanel":
    """Build a Judge or JudgePanel from config.

    Supports both legacy single-judge config and new multi-judge panel config.
    Returns a JudgePanel if 'judges' key exists in config, otherwise a Judge.
    """
    from .models import build_client_from_dict

    if "judges" in config:
        judge_clients = []
        for judge_cfg in config["judges"]:
            client = build_client_from_dict({
                "id": judge_cfg["id"],
                "name": judge_cfg.get("name", "judge"),
                "provider": judge_cfg["provider"],
                "api_base": judge_cfg.get("api_base"),
                "temperature": judge_cfg.get("temperature", 0.0),
                "max_tokens": config.get("settings", {}).get(
                    "judge_max_tokens", 1024
                ),
            })
            judge_clients.append(client)
        return JudgePanel(
            judge_clients=judge_clients,
            aggregation=config.get("judge_aggregation", "median"),
            randomize_rubric_order=config.get("settings", {}).get(
                "randomize_rubric_order", True
            ),
        )
    else:
        # Legacy single-judge config
        judge_cfg = config["judge"]
        client = build_client_from_dict({
            "id": judge_cfg["id"],
            "name": "judge",
            "provider": judge_cfg["provider"],
            "api_base": judge_cfg.get("api_base"),
            "temperature": judge_cfg.get("temperature", 0.0),
            "max_tokens": config.get("settings", {}).get(
                "judge_max_tokens", 1024
            ),
        })
        return Judge(client)
