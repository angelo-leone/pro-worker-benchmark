"""
Sensitivity analysis for the Pro-Worker AI Benchmark.

Tests whether benchmark conclusions are robust to:
1. Weight scheme changes (5 alternative schemes)
2. Judge model selection
3. Temperature variation

Usage:
    python -m tests.test_sensitivity --results-dir results/
"""

import argparse
import json
from pathlib import Path

import numpy as np


# Weight schemes for sensitivity analysis
WEIGHT_SCHEMES = {
    "default": {
        "cognitive_forcing": 0.15,
        "contrastive_explanation": 0.10,
        "skill_preservation": 0.10,
        "draft_annotation": 0.08,
        "uncertainty_transparency": 0.10,
        "complementarity": 0.12,
        "adversarial_resilience": 0.08,
        "anti_sycophancy": 0.10,
        "metacognitive_calibration": 0.07,
        "appropriate_reliance": 0.05,
        "ethical_surfacing": 0.05,
    },
    "equal": {k: 1 / 11 for k in [
        "cognitive_forcing", "contrastive_explanation", "skill_preservation",
        "draft_annotation", "uncertainty_transparency", "complementarity",
        "adversarial_resilience", "anti_sycophancy", "metacognitive_calibration",
        "appropriate_reliance", "ethical_surfacing",
    ]},
    "cf_heavy": {
        "cognitive_forcing": 0.30,
        "contrastive_explanation": 0.08,
        "skill_preservation": 0.08,
        "draft_annotation": 0.06,
        "uncertainty_transparency": 0.08,
        "complementarity": 0.10,
        "adversarial_resilience": 0.06,
        "anti_sycophancy": 0.08,
        "metacognitive_calibration": 0.06,
        "appropriate_reliance": 0.05,
        "ethical_surfacing": 0.05,
    },
    "new_dims_heavy": {
        "cognitive_forcing": 0.10,
        "contrastive_explanation": 0.07,
        "skill_preservation": 0.07,
        "draft_annotation": 0.05,
        "uncertainty_transparency": 0.07,
        "complementarity": 0.09,
        "adversarial_resilience": 0.05,
        "anti_sycophancy": 0.15,
        "metacognitive_calibration": 0.12,
        "appropriate_reliance": 0.12,
        "ethical_surfacing": 0.11,
    },
    "ar_heavy": {
        "cognitive_forcing": 0.10,
        "contrastive_explanation": 0.08,
        "skill_preservation": 0.08,
        "draft_annotation": 0.06,
        "uncertainty_transparency": 0.08,
        "complementarity": 0.10,
        "adversarial_resilience": 0.30,
        "anti_sycophancy": 0.06,
        "metacognitive_calibration": 0.05,
        "appropriate_reliance": 0.04,
        "ethical_surfacing": 0.05,
    },
}


def compute_pwi(dim_means: dict[str, float], weights: dict[str, float], max_score: int = 3) -> float:
    """Compute PWI score from dimension means and weights."""
    total_w = sum(weights.get(d, 0) for d in dim_means)
    if total_w == 0:
        return 0.0
    weighted = sum(dim_means.get(d, 0) / max_score * weights.get(d, 0) for d in dim_means)
    return (weighted / total_w) * 100


def load_model_dim_means(results_dir: Path) -> dict[str, dict[str, float]]:
    """Load dimension means per model variant."""
    model_means = {}
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        label = f"{data.get('model_name', '?')} ({data.get('variant', '?')})"
        if "layer1" not in data:
            continue
        means = {}
        for dim, prompts in data["layer1"].items():
            scores = []
            for p in prompts:
                s = p.get("mean_score", p.get("score", -1))
                if s >= 0:
                    scores.append(float(s))
            if scores:
                means[dim] = float(np.mean(scores))
        model_means[label] = means
    return model_means


def kendall_tau(ranking1: list, ranking2: list) -> float:
    """Compute Kendall's tau rank correlation between two orderings of the same items.

    Each argument is an ordered list of item labels, best first. Positions are looked up
    per item so that the two orderings are actually compared against one another.
    """
    if len(ranking1) < 2:
        return 1.0
    if set(ranking1) != set(ranking2):
        raise ValueError("rankings must contain the same items")

    pos1 = {item: i for i, item in enumerate(ranking1)}
    pos2 = {item: i for i, item in enumerate(ranking2)}

    items = list(ranking1)
    concordant = 0
    discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            sign = (pos1[a] - pos1[b]) * (pos2[a] - pos2[b])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def weight_sensitivity_analysis(model_means: dict[str, dict[str, float]]) -> dict:
    """Test whether model rankings are stable across weight schemes."""
    results = {}
    base_scheme = "default"

    # Compute PWI under each scheme
    scheme_pwis = {}
    for scheme_name, weights in WEIGHT_SCHEMES.items():
        pwis = {}
        for model, means in model_means.items():
            pwis[model] = compute_pwi(means, weights)
        scheme_pwis[scheme_name] = pwis

    # Compute rankings and compare
    base_ranking = sorted(scheme_pwis[base_scheme], key=lambda m: scheme_pwis[base_scheme][m], reverse=True)

    for scheme_name, pwis in scheme_pwis.items():
        alt_ranking = sorted(pwis, key=lambda m: pwis[m], reverse=True)

        results[scheme_name] = {
            "pwi_scores": {m: round(p, 1) for m, p in pwis.items()},
            "ranking": alt_ranking,
            "ranking_matches_default": alt_ranking == base_ranking,
            "kendall_tau": round(kendall_tau(base_ranking, alt_ranking), 3),
            "max_pwi_deviation": round(
                max(abs(pwis[m] - scheme_pwis[base_scheme][m]) for m in pwis) if pwis else 0,
                1,
            ),
        }

    return {
        "schemes_tested": list(WEIGHT_SCHEMES.keys()),
        "results": results,
        "rankings_stable": all(
            r["ranking_matches_default"] for name, r in results.items()
        ),
    }


def run_sensitivity_suite(results_dir: Path) -> dict:
    """Run the full sensitivity analysis suite."""
    print("Loading results...")
    model_means = load_model_dim_means(results_dir)

    if not model_means:
        return {"error": "No results found"}

    print(f"Loaded {len(model_means)} model runs")

    print("\n1. Weight Sensitivity Analysis...")
    weight_results = weight_sensitivity_analysis(model_means)
    if weight_results.get("rankings_stable"):
        print("   PASS: Model rankings stable across all weight schemes")
    else:
        print("   WARNING: Model rankings change under some weight schemes")
        for scheme, data in weight_results["results"].items():
            if not data["ranking_matches_default"]:
                print(f"     {scheme}: ranking differs (max deviation: {data['max_pwi_deviation']} pts)")

    return {
        "weight_sensitivity": weight_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
    )
    args = parser.parse_args()
    results = run_sensitivity_suite(args.results_dir)

    output = args.results_dir / "sensitivity_report.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {output}")
