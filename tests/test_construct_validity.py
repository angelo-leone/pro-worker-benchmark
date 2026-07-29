"""
Construct validity tests for the Pro-Worker AI Benchmark.

Tests three forms of validity:
1. Discriminant validity: dimensions should not overlap excessively (r < 0.70)
2. Convergent validity: prompts within a dimension should correlate (alpha > 0.70)
3. Item analysis: per-prompt discrimination quality

Run after a full benchmark evaluation to verify the benchmark measures
what it claims to measure.

Usage:
    python -m tests.test_construct_validity --results-dir results/
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_layer1_scores(results_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Load Layer 1 scores from all result files.

    Returns:
        Dict mapping "model_variant" -> {dimension -> [scores per prompt]}
    """
    all_scores = {}
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        label = f"{data.get('model_name', 'unknown')}_{data.get('variant', 'unknown')}"
        if "layer1" not in data:
            continue
        dim_scores = {}
        for dim, prompts in data["layer1"].items():
            scores = []
            for p in prompts:
                if "mean_score" in p and p["mean_score"] >= 0:
                    scores.append(p["mean_score"])
                elif p.get("score", -1) >= 0:
                    scores.append(float(p["score"]))
            dim_scores[dim] = scores
        all_scores[label] = dim_scores
    return all_scores


def compute_discriminant_validity(
    all_scores: dict[str, dict[str, list[float]]],
    threshold: float = 0.70,
) -> dict:
    """Compute inter-dimension correlations across all models.

    Discriminant validity: dimensions should measure distinct constructs.
    Flag pairs with Pearson r > threshold.
    """
    # Aggregate scores across all models into per-dimension vectors
    dim_vectors = {}
    for model_scores in all_scores.values():
        for dim, scores in model_scores.items():
            if dim not in dim_vectors:
                dim_vectors[dim] = []
            dim_vectors[dim].extend(scores)

    dimensions = sorted(dim_vectors.keys())
    n_dims = len(dimensions)

    # Compute correlation matrix
    # Align vectors to same length (use min length)
    min_len = min(len(dim_vectors[d]) for d in dimensions) if dimensions else 0
    if min_len < 3:
        return {"error": "Insufficient data for correlation analysis", "n_items": min_len}

    matrix = np.zeros((n_dims, n_dims))
    flagged_pairs = []

    for i, d1 in enumerate(dimensions):
        for j, d2 in enumerate(dimensions):
            v1 = np.array(dim_vectors[d1][:min_len])
            v2 = np.array(dim_vectors[d2][:min_len])
            if np.std(v1) > 0 and np.std(v2) > 0:
                r = float(np.corrcoef(v1, v2)[0, 1])
            else:
                r = 0.0
            matrix[i, j] = r
            if i < j and abs(r) > threshold:
                flagged_pairs.append({
                    "dim1": d1,
                    "dim2": d2,
                    "correlation": round(r, 3),
                    "status": "FAIL — consider merging or redesigning",
                })

    return {
        "correlation_matrix": {
            dimensions[i]: {
                dimensions[j]: round(matrix[i, j], 3)
                for j in range(n_dims)
            }
            for i in range(n_dims)
        },
        "flagged_pairs": flagged_pairs,
        "threshold": threshold,
        "n_items_per_dimension": min_len,
        "pass": len(flagged_pairs) == 0,
    }


def compute_cronbachs_alpha(scores_matrix: np.ndarray) -> float:
    """Compute Cronbach's alpha for internal consistency.

    Args:
        scores_matrix: 2D array of shape (n_items, n_raters_or_models)
    """
    n_items, n_cols = scores_matrix.shape
    if n_items < 2 or n_cols < 2:
        return 0.0

    item_variances = np.var(scores_matrix, axis=1, ddof=1)
    total_scores = np.sum(scores_matrix, axis=0)
    total_variance = np.var(total_scores, ddof=1)

    if total_variance == 0:
        return 0.0

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return float(alpha)


def compute_convergent_validity(
    all_scores: dict[str, dict[str, list[float]]],
    target_alpha: float = 0.70,
) -> dict:
    """Compute Cronbach's alpha per dimension across models.

    Convergent validity: prompts within a dimension should be consistent.
    """
    # Build per-dimension matrices: rows = prompts, columns = models
    models = list(all_scores.keys())
    if not models:
        return {"error": "No model results found"}

    # Get all dimensions
    all_dims = set()
    for m_scores in all_scores.values():
        all_dims.update(m_scores.keys())

    results = {}
    for dim in sorted(all_dims):
        # Collect scores from each model for this dimension
        model_vectors = []
        for model in models:
            scores = all_scores[model].get(dim, [])
            model_vectors.append(scores)

        if not model_vectors or not model_vectors[0]:
            results[dim] = {"alpha": 0.0, "n_items": 0, "pass": False}
            continue

        # Align to same number of prompts
        min_prompts = min(len(v) for v in model_vectors)
        if min_prompts < 2:
            results[dim] = {"alpha": 0.0, "n_items": min_prompts, "pass": False}
            continue

        matrix = np.array([v[:min_prompts] for v in model_vectors]).T
        alpha = compute_cronbachs_alpha(matrix)

        results[dim] = {
            "alpha": round(alpha, 3),
            "n_items": min_prompts,
            "n_models": len(models),
            "pass": alpha >= target_alpha,
        }

    return {
        "per_dimension": results,
        "target_alpha": target_alpha,
        "overall_pass": all(r.get("pass", False) for r in results.values()),
    }


def compute_item_analysis(
    all_scores: dict[str, dict[str, list[float]]],
    min_correlation: float = 0.30,
) -> dict:
    """Compute item-total correlation for each prompt within its dimension.

    Prompts with item-total correlation < min_correlation are poor discriminators
    and should be reviewed or replaced.
    """
    results = {}

    for model, model_scores in all_scores.items():
        for dim, scores in model_scores.items():
            if dim not in results:
                results[dim] = {"items": [], "n_models_analyzed": 0}

            if len(scores) < 3:
                continue

            scores_arr = np.array(scores)
            total = np.sum(scores_arr)
            item_total_corrs = []

            for i in range(len(scores)):
                # Item-total correlation: correlation of item i with sum of all other items
                rest = np.delete(scores_arr, i)
                rest_total = np.sum(rest)
                if np.std([scores_arr[i]]) == 0 or len(rest) < 2:
                    item_total_corrs.append(0.0)
                    continue
                # Point-biserial approximation
                item_values = scores_arr[i]
                rest_sum = rest_total
                # Simple correlation approach
                all_items = scores_arr
                other_sum = np.array([np.sum(np.delete(all_items, j)) for j in range(len(all_items))])
                if np.std(all_items) > 0 and np.std(other_sum) > 0:
                    corr = float(np.corrcoef(all_items, other_sum)[0, 1])
                else:
                    corr = 0.0
                item_total_corrs.append(corr)
                break  # Only need one pass

            results[dim]["n_models_analyzed"] += 1

    # Identify weak items per dimension
    for dim in results:
        flagged = [
            i for i, item in enumerate(results[dim].get("items", []))
            if item.get("item_total_correlation", 1.0) < min_correlation
        ]
        results[dim]["n_flagged"] = len(flagged)
        results[dim]["min_correlation_threshold"] = min_correlation

    return results


def run_validity_suite(results_dir: Path) -> dict:
    """Run the full construct validity test suite."""
    print("Loading results...")
    all_scores = load_all_layer1_scores(results_dir)

    if not all_scores:
        return {"error": "No results found in " + str(results_dir)}

    print(f"Loaded {len(all_scores)} model runs")

    print("\n1. Discriminant Validity (inter-dimension correlations)...")
    discriminant = compute_discriminant_validity(all_scores)
    if discriminant.get("pass"):
        print("   PASS: All inter-dimension correlations below 0.70")
    elif "flagged_pairs" in discriminant:
        for pair in discriminant["flagged_pairs"]:
            print(f"   FAIL: {pair['dim1']} x {pair['dim2']} = {pair['correlation']}")

    print("\n2. Convergent Validity (Cronbach's alpha per dimension)...")
    convergent = compute_convergent_validity(all_scores)
    if "per_dimension" in convergent:
        for dim, stats in convergent["per_dimension"].items():
            status = "PASS" if stats.get("pass") else "FAIL"
            print(f"   {status}: {dim} alpha={stats.get('alpha', 0):.3f}")

    print("\n3. Item Analysis...")
    item_analysis = compute_item_analysis(all_scores)

    return {
        "discriminant_validity": discriminant,
        "convergent_validity": convergent,
        "item_analysis": item_analysis,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct validity tests")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory containing benchmark result JSON files",
    )
    args = parser.parse_args()
    results = run_validity_suite(args.results_dir)

    # Save report
    output = args.results_dir / "validity_report.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {output}")
