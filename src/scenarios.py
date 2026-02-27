"""
Multi-turn scenario handler for the Pro-Worker AI Benchmark.
Manages conversations across multiple turns, tracking context and scoring each turn.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import ModelClient, ModelResponse
from .judge import Judge


@dataclass
class ScenarioTurn:
    """A single turn in a multi-turn scenario."""
    turn_number: int
    user_message: str
    expected_behaviors: dict[str, bool]
    user_continued: str = ""


@dataclass
class Scenario:
    """A multi-turn evaluation scenario."""
    scenario_id: str
    domain: str
    user_persona: str
    dimensions_tested: list[str]
    turns: list[ScenarioTurn]


@dataclass
class TurnResult:
    """Result of evaluating a single turn."""
    turn_number: int
    user_message: str
    ai_response: str
    dimension_scores: dict[str, dict]
    behavior_results: dict[str, bool | None]
    latency_ms: float


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""
    scenario_id: str
    model_id: str
    with_system_prompt: bool
    turn_results: list[TurnResult]
    average_score: float = 0.0
    behavior_pass_rate: float = 0.0


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    turns = []
    for turn_data in data["turns"]:
        turns.append(ScenarioTurn(
            turn_number=turn_data["turn"],
            user_message=turn_data["user"],
            expected_behaviors=turn_data.get("expected_behaviors", {}),
            user_continued=turn_data.get("user_continued", ""),
        ))

    return Scenario(
        scenario_id=data["scenario_id"],
        domain=data["domain"],
        user_persona=data["user_persona"],
        dimensions_tested=data["dimensions_tested"],
        turns=turns,
    )


def load_all_scenarios(scenarios_dir: Path) -> list[Scenario]:
    """Load all scenario files from a directory."""
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios


class ScenarioRunner:
    """Runs multi-turn scenarios against a model and evaluates each turn."""

    def __init__(
        self,
        model_client: ModelClient,
        judge: Judge,
        system_prompt: str | None = None,
    ):
        self.model = model_client
        self.judge = judge
        self.system_prompt = system_prompt

    def run(self, scenario: Scenario) -> ScenarioResult:
        """
        Execute a full multi-turn scenario.
        Builds up conversation history turn by turn, scoring each turn.
        """
        conversation_history: list[dict[str, str]] = []
        turn_results: list[TurnResult] = []

        for turn in scenario.turns:
            # Build the user message (may have continuation)
            user_msg = turn.user_message
            if turn.user_continued:
                user_msg += "\n" + turn.user_continued

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_msg})

            # Call the model
            response = self.model.call(
                messages=conversation_history,
                system_prompt=self.system_prompt,
            )

            if response.error:
                # Record error and continue
                turn_results.append(TurnResult(
                    turn_number=turn.turn_number,
                    user_message=user_msg,
                    ai_response=f"[ERROR: {response.error}]",
                    dimension_scores={},
                    behavior_results={k: None for k in turn.expected_behaviors},
                    latency_ms=response.latency_ms,
                ))
                # Still add to history so context builds up
                conversation_history.append({
                    "role": "assistant",
                    "content": "[Error generating response]",
                })
                continue

            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            # Build prior turns for judge context
            prior_turns = []
            for i in range(0, len(conversation_history) - 2, 2):
                if i + 1 < len(conversation_history):
                    prior_turns.append({
                        "user": conversation_history[i]["content"],
                        "assistant": conversation_history[i + 1]["content"],
                    })

            # Score this turn
            judge_result = self.judge.score_scenario_turn(
                dimensions=scenario.dimensions_tested,
                user_prompt=user_msg,
                ai_response=response.content,
                expected_behaviors=turn.expected_behaviors,
                prior_turns=prior_turns[:-1] if prior_turns else None,
                context=f"Scenario: {scenario.scenario_id}, Persona: {scenario.user_persona}",
            )

            # Extract dimension scores and behaviors
            behaviors = judge_result.pop("behaviors", {})
            dimension_scores = judge_result

            turn_results.append(TurnResult(
                turn_number=turn.turn_number,
                user_message=user_msg,
                ai_response=response.content,
                dimension_scores=dimension_scores,
                behavior_results=behaviors,
                latency_ms=response.latency_ms,
            ))

        # Compute aggregate scores
        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            model_id=self.model.config.id,
            with_system_prompt=self.system_prompt is not None,
            turn_results=turn_results,
        )
        result.average_score = self._compute_average_score(turn_results)
        result.behavior_pass_rate = self._compute_behavior_pass_rate(turn_results)
        return result

    def _compute_average_score(self, turn_results: list[TurnResult]) -> float:
        """Compute average dimension score across all turns."""
        all_scores = []
        for tr in turn_results:
            for dim, result in tr.dimension_scores.items():
                score = result.get("score", -1)
                if score >= 0:
                    all_scores.append(score)
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def _compute_behavior_pass_rate(self, turn_results: list[TurnResult]) -> float:
        """Compute percentage of expected behaviors that were exhibited."""
        total = 0
        passed = 0
        for tr in turn_results:
            for name, exhibited in tr.behavior_results.items():
                if exhibited is not None:
                    total += 1
                    if exhibited:
                        passed += 1
        return passed / total if total > 0 else 0.0
