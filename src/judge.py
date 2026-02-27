"""
LLM-as-Judge scoring module for the Pro-Worker AI Benchmark.
Uses a separate LLM instance to evaluate responses against rubrics.
"""

import json
import re
from pathlib import Path

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


def _build_rubric_text(dimension: str, rubrics: dict) -> str:
    """Format the rubric for a dimension into judge-readable text."""
    dim = rubrics[dimension]
    lines = [
        f"DIMENSION: {dim['name']}",
        f"DESCRIPTION: {dim['description']}",
        "",
        "SCORING RUBRIC:",
    ]
    for score in [3, 2, 1, 0]:
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
    ) -> dict:
        """
        Score an AI response on a specific dimension.

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
            'and "evidence" (specific quote or description).'
        )

        eval_prompt = "\n".join(parts)

        # Call the judge model
        response = self.client.call(
            messages=[{"role": "user", "content": eval_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.0,
        )

        if response.error:
            return {
                "score": -1,
                "reasoning": f"Judge error: {response.error}",
                "evidence": "",
                "raw_judge_response": "",
            }

        parsed = _parse_judge_response(response.content)
        parsed["raw_judge_response"] = response.content
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
