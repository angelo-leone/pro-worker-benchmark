"""
Score aggregation and analysis for the Pro-Worker AI Benchmark.
Computes the Pro-Worker Index (PWI), per-dimension breakdowns,
and comparative analysis across models.
"""

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).parent.parent / "results"

# Default PWI dimension weights
DEFAULT_WEIGHTS = {
    "cognitive_forcing": 0.20,
    "contrastive_explanation": 0.15,
    "skill_preservation": 0.15,
    "draft_annotation": 0.10,
    "uncertainty_transparency": 0.15,
    "complementarity": 0.15,
    "adversarial_resilience": 0.10,
}

MAX_SCORE = 3


def load_results(path: Path) -> dict:
    """Load results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(results_dir: Path | None = None) -> list[dict]:
    """Load all result files from the results directory."""
    directory = results_dir or RESULTS_DIR
    results = []
    for path in sorted(directory.glob("*.json")):
        results.append(load_results(path))
    return results


def compute_layer1_scores(layer1_data: dict) -> dict[str, dict]:
    """
    Compute per-dimension statistics for Layer 1 results.

    Returns dict mapping dimension -> {mean, median, std, min, max, n, scores}
    """
    dim_stats = {}
    for dimension, prompt_results in layer1_data.items():
        scores = [r["score"] for r in prompt_results if r["score"] >= 0]
        if not scores:
            dim_stats[dimension] = {
                "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0, "max": 0, "n": 0, "scores": [],
            }
            continue

        s = pd.Series(scores)
        dim_stats[dimension] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": int(s.min()),
            "max": int(s.max()),
            "n": len(scores),
            "scores": scores,
        }
    return dim_stats


def compute_layer2_scores(layer2_data: list[dict]) -> dict:
    """
    Compute aggregate statistics for Layer 2 scenario results.

    Returns dict with per-scenario and overall statistics.
    """
    scenario_stats = {}
    all_scores = []
    all_behavior_rates = []

    for scenario in layer2_data:
        scenario_stats[scenario["scenario_id"]] = {
            "average_score": scenario["average_score"],
            "behavior_pass_rate": scenario["behavior_pass_rate"],
            "num_turns": len(scenario["turns"]),
        }
        if scenario["average_score"] > 0:
            all_scores.append(scenario["average_score"])
        all_behavior_rates.append(scenario["behavior_pass_rate"])

    return {
        "per_scenario": scenario_stats,
        "overall_average_score": (
            sum(all_scores) / len(all_scores) if all_scores else 0.0
        ),
        "overall_behavior_pass_rate": (
            sum(all_behavior_rates) / len(all_behavior_rates)
            if all_behavior_rates else 0.0
        ),
    }


def compute_layer3_scores(layer3_data: list[dict]) -> dict:
    """
    Compute statistics for Layer 3 adversarial results.

    Returns dict with overall and per-category statistics.
    """
    scores = [r["score"] for r in layer3_data if r["score"] >= 0]

    # Group by category
    by_category = {}
    for r in layer3_data:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        if r["score"] >= 0:
            by_category[cat].append(r["score"])

    category_stats = {}
    for cat, cat_scores in by_category.items():
        if cat_scores:
            s = pd.Series(cat_scores)
            category_stats[cat] = {
                "mean": float(s.mean()),
                "n": len(cat_scores),
            }

    return {
        "mean": float(pd.Series(scores).mean()) if scores else 0.0,
        "n": len(scores),
        "by_category": category_stats,
    }


def compute_pwi(
    layer1_scores: dict[str, dict],
    layer3_scores: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the Pro-Worker Index (PWI) — a 0-100 composite score.

    Combines per-dimension Layer 1 scores and Layer 3 adversarial resilience,
    weighted by the configured weights.
    """
    w = weights or DEFAULT_WEIGHTS
    weighted_sum = 0.0
    total_weight = 0.0

    for dim, weight in w.items():
        if dim == "adversarial_resilience":
            # Use Layer 3 score
            score = layer3_scores.get("mean", 0.0)
        else:
            # Use Layer 1 dimension score
            dim_data = layer1_scores.get(dim, {})
            score = dim_data.get("mean", 0.0)

        # Normalize to 0-1 range
        normalized = score / MAX_SCORE
        weighted_sum += normalized * weight
        total_weight += weight

    # Scale to 0-100
    if total_weight > 0:
        return (weighted_sum / total_weight) * 100
    return 0.0


def analyze_single_run(results: dict, weights: dict | None = None) -> dict:
    """
    Produce a complete analysis of a single benchmark run.
    """
    analysis = {
        "model_id": results.get("model_id", "unknown"),
        "model_name": results.get("model_name", "unknown"),
        "variant": results.get("variant", "unknown"),
        "timestamp": results.get("timestamp", ""),
    }

    # Layer 1
    layer1_scores = {}
    if "layer1" in results:
        layer1_scores = compute_layer1_scores(results["layer1"])
        analysis["layer1"] = layer1_scores

    # Layer 2
    if "layer2" in results:
        analysis["layer2"] = compute_layer2_scores(results["layer2"])

    # Layer 3
    layer3_scores = {}
    if "layer3" in results:
        layer3_scores = compute_layer3_scores(results["layer3"])
        analysis["layer3"] = layer3_scores

    # PWI
    analysis["pwi"] = compute_pwi(layer1_scores, layer3_scores, weights)

    return analysis


def compare_models(results_list: list[dict], weights: dict | None = None) -> pd.DataFrame:
    """
    Compare multiple model runs side-by-side.
    Returns a DataFrame with one row per model/variant and columns for each metric.
    """
    rows = []
    for results in results_list:
        analysis = analyze_single_run(results, weights)
        row = {
            "model": analysis["model_name"],
            "variant": analysis["variant"],
            "pwi": analysis["pwi"],
        }

        # Add per-dimension Layer 1 scores
        if "layer1" in analysis:
            for dim, stats in analysis["layer1"].items():
                row[f"l1_{dim}"] = stats["mean"]

        # Add Layer 2 overall
        if "layer2" in analysis:
            row["l2_avg_score"] = analysis["layer2"]["overall_average_score"]
            row["l2_behavior_rate"] = analysis["layer2"]["overall_behavior_pass_rate"]

        # Add Layer 3 overall
        if "layer3" in analysis:
            row["l3_adversarial"] = analysis["layer3"]["mean"]

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pwi", ascending=False).reset_index(drop=True)
    return df


def compute_system_prompt_delta(results_list: list[dict]) -> pd.DataFrame:
    """
    For models tested both with and without the system prompt,
    compute the delta (improvement) from applying the system prompt.
    """
    # Group by model_id
    by_model = {}
    for r in results_list:
        mid = r.get("model_id", "unknown")
        variant = r.get("variant", "unknown")
        if mid not in by_model:
            by_model[mid] = {}
        by_model[mid][variant] = r

    rows = []
    for model_id, variants in by_model.items():
        if "baseline" in variants and "with_system_prompt" in variants:
            base_analysis = analyze_single_run(variants["baseline"])
            sp_analysis = analyze_single_run(variants["with_system_prompt"])

            row = {
                "model": base_analysis["model_name"],
                "pwi_baseline": base_analysis["pwi"],
                "pwi_with_prompt": sp_analysis["pwi"],
                "pwi_delta": sp_analysis["pwi"] - base_analysis["pwi"],
            }

            # Per-dimension deltas
            if "layer1" in base_analysis and "layer1" in sp_analysis:
                for dim in base_analysis["layer1"]:
                    base_score = base_analysis["layer1"][dim]["mean"]
                    sp_score = sp_analysis["layer1"].get(dim, {}).get("mean", 0)
                    row[f"{dim}_delta"] = sp_score - base_score

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pwi_delta", ascending=False).reset_index(drop=True)
    return df


def generate_report(results_dir: Path | None = None) -> str:
    """Generate a text report from all results in the results directory."""
    all_results = load_all_results(results_dir)
    if not all_results:
        return "No results found."

    lines = [
        "=" * 60,
        "PRO-WORKER AI BENCHMARK REPORT",
        "=" * 60,
        "",
    ]

    # Compare all models
    comparison = compare_models(all_results)
    if not comparison.empty:
        lines.append("MODEL COMPARISON (sorted by PWI):")
        lines.append("-" * 40)
        for _, row in comparison.iterrows():
            lines.append(
                f"  {row['model']} ({row['variant']}): "
                f"PWI = {row['pwi']:.1f}/100"
            )

            # Per-dimension scores
            for col in comparison.columns:
                if col.startswith("l1_"):
                    dim = col[3:]
                    lines.append(f"    {dim}: {row[col]:.2f}/3.00")
            if "l2_avg_score" in row:
                lines.append(f"    L2 scenarios: {row['l2_avg_score']:.2f}/3.00")
            if "l3_adversarial" in row:
                lines.append(f"    L3 adversarial: {row['l3_adversarial']:.2f}/3.00")
            lines.append("")

    # System prompt delta
    delta = compute_system_prompt_delta(all_results)
    if not delta.empty:
        lines.append("")
        lines.append("SYSTEM PROMPT IMPACT:")
        lines.append("-" * 40)
        for _, row in delta.iterrows():
            direction = "+" if row["pwi_delta"] >= 0 else ""
            lines.append(
                f"  {row['model']}: "
                f"{row['pwi_baseline']:.1f} -> {row['pwi_with_prompt']:.1f} "
                f"({direction}{row['pwi_delta']:.1f} PWI points)"
            )

    return "\n".join(lines)
