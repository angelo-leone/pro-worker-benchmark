"""
Main benchmark runner for the Pro-Worker AI Benchmark.
Orchestrates prompt loading, model calls, judging, and result saving.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .models import ModelClient, ModelConfig, build_client_from_dict
from .judge import Judge
from .scenarios import ScenarioRunner, load_all_scenarios


BASE_DIR = Path(__file__).parent.parent

# Load .env file for API keys
load_dotenv(BASE_DIR / ".env")
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"


def load_config(config_path: Path | None = None) -> dict:
    """Load benchmark configuration."""
    path = config_path or (BASE_DIR / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(config: dict) -> str | None:
    """Load the pro-worker system prompt if configured."""
    if not config["settings"].get("test_with_system_prompt"):
        return None
    sp_path = BASE_DIR / config["settings"]["system_prompt_path"]
    if sp_path.exists():
        with open(sp_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def load_layer1_prompts() -> dict[str, list[dict]]:
    """Load all Layer 1 behavioral probe prompts, grouped by dimension."""
    prompts_by_dimension = {}
    layer1_dir = PROMPTS_DIR / "layer1_behavioral"
    for path in sorted(layer1_dir.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dimension = data["dimension"]
        prompts_by_dimension[dimension] = data["prompts"]
    return prompts_by_dimension


def load_layer3_prompts() -> list[dict]:
    """Load all Layer 3 adversarial stress test prompts."""
    path = PROMPTS_DIR / "layer3_adversarial" / "stress_tests.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompts"]


def run_layer1(
    model_client: ModelClient,
    judge: Judge,
    prompts_by_dimension: dict[str, list[dict]],
    system_prompt: str | None = None,
) -> dict:
    """
    Run Layer 1: Behavioral Probes.
    Each prompt is sent to the model, and the response is scored by the judge.
    """
    results = {}

    for dimension, prompts in prompts_by_dimension.items():
        dim_results = []
        desc = f"Layer 1 - {dimension}"
        for prompt_data in tqdm(prompts, desc=desc, leave=False):
            prompt_text = prompt_data["prompt"]

            # Build messages, including prior_turns if present
            messages = []
            prior_turns = prompt_data.get("prior_turns", [])
            for pt in prior_turns:
                messages.append({"role": "user", "content": pt["user"]})
                messages.append({"role": "assistant", "content": pt["assistant"]})
            messages.append({"role": "user", "content": prompt_text})

            # Call model
            response = model_client.call(
                messages=messages,
                system_prompt=system_prompt,
            )

            if response.error:
                dim_results.append({
                    "prompt_id": prompt_data["id"],
                    "prompt": prompt_text,
                    "response": "",
                    "error": response.error,
                    "score": -1,
                    "reasoning": f"Model error: {response.error}",
                    "evidence": "",
                    "latency_ms": response.latency_ms,
                })
                continue

            # Judge the response
            judge_result = judge.score(
                dimension=dimension,
                user_prompt=prompt_text,
                ai_response=response.content,
                context=prompt_data.get("context", ""),
                prior_turns=prior_turns if prior_turns else None,
            )

            dim_results.append({
                "prompt_id": prompt_data["id"],
                "prompt": prompt_text,
                "response": response.content,
                "score": judge_result["score"],
                "reasoning": judge_result["reasoning"],
                "evidence": judge_result["evidence"],
                "latency_ms": response.latency_ms,
            })

        results[dimension] = dim_results
    return results


def run_layer2(
    model_client: ModelClient,
    judge: Judge,
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Run Layer 2: Multi-Turn Scenarios.
    """
    scenarios_dir = PROMPTS_DIR / "layer2_scenarios"
    scenarios = load_all_scenarios(scenarios_dir)
    runner = ScenarioRunner(model_client, judge, system_prompt)

    results = []
    for scenario in tqdm(scenarios, desc="Layer 2 - Scenarios", leave=False):
        scenario_result = runner.run(scenario)

        # Serialize turn results
        serialized_turns = []
        for tr in scenario_result.turn_results:
            serialized_turns.append({
                "turn": tr.turn_number,
                "user_message": tr.user_message,
                "ai_response": tr.ai_response,
                "dimension_scores": {
                    dim: {
                        "score": scores.get("score", -1),
                        "reasoning": scores.get("reasoning", ""),
                    }
                    for dim, scores in tr.dimension_scores.items()
                },
                "behavior_results": tr.behavior_results,
                "latency_ms": tr.latency_ms,
            })

        results.append({
            "scenario_id": scenario_result.scenario_id,
            "model_id": scenario_result.model_id,
            "with_system_prompt": scenario_result.with_system_prompt,
            "turns": serialized_turns,
            "average_score": scenario_result.average_score,
            "behavior_pass_rate": scenario_result.behavior_pass_rate,
        })

    return results


def run_layer3(
    model_client: ModelClient,
    judge: Judge,
    prompts: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Run Layer 3: Adversarial Stress Tests.
    """
    results = []
    for prompt_data in tqdm(prompts, desc="Layer 3 - Adversarial", leave=False):
        prompt_text = prompt_data["prompt"]

        # Some adversarial prompts have prior context
        messages = []
        if prompt_data.get("prior_context"):
            messages.append({
                "role": "system",
                "content": f"Context: {prompt_data['prior_context']}",
            })
        messages.append({"role": "user", "content": prompt_text})

        # Add any supplementary data
        if prompt_data.get("data"):
            messages[-1]["content"] += f"\n\nData: {prompt_data['data']}"
        if prompt_data.get("spec"):
            messages[-1]["content"] += f"\n\nSpec: {prompt_data['spec']}"
        if prompt_data.get("task"):
            messages[-1]["content"] += f"\n\nTask: {prompt_data['task']}"

        response = model_client.call(
            messages=messages,
            system_prompt=system_prompt,
        )

        if response.error:
            results.append({
                "prompt_id": prompt_data["id"],
                "category": prompt_data.get("category", "unknown"),
                "prompt": prompt_text,
                "response": "",
                "error": response.error,
                "score": -1,
                "reasoning": f"Model error: {response.error}",
                "latency_ms": response.latency_ms,
            })
            continue

        judge_result = judge.score(
            dimension="adversarial_resilience",
            user_prompt=prompt_text,
            ai_response=response.content,
            context=prompt_data.get("context", ""),
        )

        results.append({
            "prompt_id": prompt_data["id"],
            "category": prompt_data.get("category", "unknown"),
            "prompt": prompt_text,
            "response": response.content,
            "score": judge_result["score"],
            "reasoning": judge_result["reasoning"],
            "evidence": judge_result.get("evidence", ""),
            "latency_ms": response.latency_ms,
        })

    return results


def save_results(results: dict, model_name: str, results_dir: Path | None = None):
    """Save benchmark results to a JSON file."""
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_").replace(":", "_")
    filename = f"{safe_name}_{timestamp}.json"

    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def run_benchmark(
    config_path: Path | None = None,
    layers: list[int] | None = None,
    models_filter: list[str] | None = None,
):
    """
    Run the full benchmark.

    Args:
        config_path: Path to config.yaml (default: auto-detect)
        layers: Which layers to run (default: [1, 2, 3])
        models_filter: Only run these model IDs (default: all)
    """
    config = load_config(config_path)
    layers = layers or [1, 2, 3]
    system_prompt = load_system_prompt(config)

    # Build judge client
    judge_config = config["judge"]
    judge_client = build_client_from_dict({
        "id": judge_config["id"],
        "name": "judge",
        "provider": judge_config["provider"],
        "api_base": judge_config.get("api_base"),
        "temperature": judge_config.get("temperature", 0.0),
        "max_tokens": config["settings"].get("judge_max_tokens", 512),
    })
    judge = Judge(judge_client)

    # Load prompts
    layer1_prompts = load_layer1_prompts() if 1 in layers else {}
    layer3_prompts = load_layer3_prompts() if 3 in layers else []

    # Determine which models to test
    models_to_test = config["models"]
    if models_filter:
        models_to_test = [m for m in models_to_test if m["id"] in models_filter]

    all_results = {}

    # Apply config settings to model defaults
    model_max_tokens = config["settings"].get("max_tokens", 512)
    model_temperature = config["settings"].get("model_temperature", 0.7)

    for model_dict in models_to_test:
        model_name = model_dict["name"]
        model_dict_with_settings = {
            **model_dict,
            "max_tokens": model_dict.get("max_tokens", model_max_tokens),
            "temperature": model_dict.get("temperature", model_temperature),
        }
        model_client = build_client_from_dict(model_dict_with_settings)

        # Determine test variants
        variants = [("baseline", None)]
        if system_prompt:
            variants.append(("with_system_prompt", system_prompt))

        for variant_name, variant_prompt in variants:
            run_label = f"{model_name} ({variant_name})"
            print(f"\n{'='*60}")
            print(f"Running: {run_label}")
            print(f"{'='*60}")

            run_results = {
                "model_id": model_dict["id"],
                "model_name": model_name,
                "variant": variant_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Layer 1: Behavioral Probes
            if 1 in layers:
                print(f"\n--- Layer 1: Behavioral Probes ---")
                run_results["layer1"] = run_layer1(
                    model_client, judge, layer1_prompts, variant_prompt
                )

            # Layer 2: Multi-Turn Scenarios
            if 2 in layers:
                print(f"\n--- Layer 2: Multi-Turn Scenarios ---")
                run_results["layer2"] = run_layer2(
                    model_client, judge, variant_prompt
                )

            # Layer 3: Adversarial Stress Tests
            if 3 in layers:
                print(f"\n--- Layer 3: Adversarial Stress Tests ---")
                run_results["layer3"] = run_layer3(
                    model_client, judge, layer3_prompts, variant_prompt
                )

            # Save results
            save_path = save_results(
                run_results,
                f"{model_dict['id']}_{variant_name}",
            )
            print(f"\nResults saved to: {save_path}")
            all_results[run_label] = run_results

    return all_results


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pro-Worker AI Benchmark Runner")
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which layers to run (1, 2, 3)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Only run specific model IDs",
    )
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        layers=args.layers,
        models_filter=args.models,
    )

    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}")

    # Print quick summary
    for label, res in results.items():
        print(f"\n{label}:")
        if "layer1" in res:
            for dim, dim_results in res["layer1"].items():
                scores = [r["score"] for r in dim_results if r["score"] >= 0]
                avg = sum(scores) / len(scores) if scores else 0
                print(f"  L1 {dim}: {avg:.2f}/3.00 ({len(scores)} prompts)")
        if "layer2" in res:
            for scenario in res["layer2"]:
                print(
                    f"  L2 {scenario['scenario_id']}: "
                    f"avg={scenario['average_score']:.2f}/3.00, "
                    f"behaviors={scenario['behavior_pass_rate']:.0%}"
                )
        if "layer3" in res:
            scores = [r["score"] for r in res["layer3"] if r["score"] >= 0]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  L3 adversarial_resilience: {avg:.2f}/3.00 ({len(scores)} prompts)")


if __name__ == "__main__":
    main()
