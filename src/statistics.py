"""
Bayesian statistical analysis for the Pro-Worker AI Benchmark.

Uses Bayesian inference instead of CLT-based methods per Bowyer et al. (2025),
"Don't use the CLT in LLM evals with fewer than a few hundred datapoints."
CLT-based confidence intervals dramatically underestimate uncertainty in
small-sample LLM benchmarks, producing error bars that are too narrow.

This module provides:
- Dirichlet-Multinomial posterior for ordinal score distributions
- Bootstrap confidence intervals for composite PWI scores
- Cohen's d effect size for system-prompt deltas
- Inter-rater reliability metrics (Cohen's kappa, Krippendorff's alpha)
"""

from collections import Counter
from itertools import combinations

import numpy as np


def bayesian_score_estimate(
    scores: list[int],
    max_score: int = 3,
    n_samples: int = 10_000,
    hdi_prob: float = 0.95,
) -> dict:
    """Compute Bayesian posterior for a dimension's mean score.

    Uses a Dirichlet-Multinomial model (conjugate prior for categorical
    scores). The prior is a uniform Dirichlet (alpha=1 for each category),
    representing no prior preference for any score level.

    Args:
        scores: List of integer scores (0 to max_score).
        max_score: Maximum possible score.
        n_samples: Number of posterior samples to draw.
        hdi_prob: Probability mass for the Highest Density Interval.

    Returns:
        Dict with posterior_mean, hdi_lower, hdi_upper, posterior_std,
        and the score distribution.
    """
    if not scores:
        return {
            "posterior_mean": 0.0,
            "hdi_lower": 0.0,
            "hdi_upper": 0.0,
            "posterior_std": 0.0,
            "n_observations": 0,
            "score_distribution": {},
        }

    # Count observations at each score level
    counts = np.zeros(max_score + 1)
    for s in scores:
        s_int = int(round(s))
        if 0 <= s_int <= max_score:
            counts[s_int] += 1

    # Dirichlet posterior: alpha_posterior = counts + prior (uniform = 1)
    alpha_posterior = counts + 1.0

    # Draw from the posterior Dirichlet distribution
    rng = np.random.default_rng(42)
    posterior_samples = rng.dirichlet(alpha_posterior, size=n_samples)

    # Compute posterior mean scores (expected value under each sample)
    score_values = np.arange(max_score + 1)
    mean_score_samples = posterior_samples @ score_values

    # Compute HDI
    lower_pct = (1 - hdi_prob) / 2 * 100
    upper_pct = (1 + hdi_prob) / 2 * 100

    return {
        "posterior_mean": float(np.mean(mean_score_samples)),
        "hdi_lower": float(np.percentile(mean_score_samples, lower_pct)),
        "hdi_upper": float(np.percentile(mean_score_samples, upper_pct)),
        "posterior_std": float(np.std(mean_score_samples)),
        "n_observations": len(scores),
        "score_distribution": {
            int(k): int(v) for k, v in zip(score_values, counts)
        },
    }


def compute_effect_size(
    baseline_scores: list[int | float],
    prompted_scores: list[int | float],
) -> dict:
    """Compute Cohen's d effect size for system prompt delta.

    Args:
        baseline_scores: Scores without system prompt.
        prompted_scores: Scores with system prompt.

    Returns:
        Dict with cohens_d, interpretation, and descriptive statistics.
    """
    if not baseline_scores or not prompted_scores:
        return {
            "cohens_d": 0.0,
            "interpretation": "insufficient_data",
            "baseline_mean": 0.0,
            "prompted_mean": 0.0,
            "baseline_std": 0.0,
            "prompted_std": 0.0,
        }

    b = np.array(baseline_scores, dtype=float)
    p = np.array(prompted_scores, dtype=float)

    n1, n2 = len(b), len(p)
    m1, m2 = np.mean(b), np.mean(p)
    s1, s2 = np.std(b, ddof=1), np.std(p, ddof=1)

    # Pooled standard deviation
    if n1 + n2 - 2 > 0 and (s1 > 0 or s2 > 0):
        s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (m2 - m1) / s_pooled if s_pooled > 0 else 0.0
    else:
        d = 0.0

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.20:
        interpretation = "negligible"
    elif abs_d < 0.50:
        interpretation = "small"
    elif abs_d < 0.80:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohens_d": float(d),
        "interpretation": interpretation,
        "baseline_mean": float(m1),
        "prompted_mean": float(m2),
        "baseline_std": float(s1),
        "prompted_std": float(s2),
        "baseline_n": n1,
        "prompted_n": n2,
    }


def bootstrap_pwi_ci(
    dimension_scores: dict[str, list[int | float]],
    weights: dict[str, float],
    max_score: int = 3,
    n_bootstrap: int = 10_000,
    ci_prob: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for composite PWI score.

    Args:
        dimension_scores: Dict mapping dimension name to list of scores.
        weights: Dict mapping dimension name to weight (must sum to ~1.0).
        max_score: Maximum score per item.
        n_bootstrap: Number of bootstrap iterations.
        ci_prob: Confidence level for the interval.

    Returns:
        Dict with PWI mean, CI bounds, and per-dimension bootstrap means.
    """
    if not dimension_scores:
        return {"pwi_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "pwi_std": 0.0}

    rng = np.random.default_rng(42)
    pwi_samples = []

    for _ in range(n_bootstrap):
        resampled_means = {}
        for dim, scores in dimension_scores.items():
            if scores:
                boot = rng.choice(scores, size=len(scores), replace=True)
                resampled_means[dim] = float(np.mean(boot))
            else:
                resampled_means[dim] = 0.0

        # Compute weighted PWI
        total_weight = sum(weights.get(d, 0) for d in dimension_scores)
        if total_weight > 0:
            weighted_sum = sum(
                resampled_means.get(d, 0) / max_score * weights.get(d, 0)
                for d in dimension_scores
            )
            pwi = (weighted_sum / total_weight) * 100
        else:
            pwi = 0.0
        pwi_samples.append(pwi)

    pwi_arr = np.array(pwi_samples)
    lower_pct = (1 - ci_prob) / 2 * 100
    upper_pct = (1 + ci_prob) / 2 * 100

    return {
        "pwi_mean": float(np.mean(pwi_arr)),
        "ci_lower": float(np.percentile(pwi_arr, lower_pct)),
        "ci_upper": float(np.percentile(pwi_arr, upper_pct)),
        "pwi_std": float(np.std(pwi_arr)),
    }


def cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's kappa for two raters on ordinal data.

    Args:
        rater1: Scores from rater 1.
        rater2: Scores from rater 2 (same length as rater1).

    Returns:
        Cohen's kappa coefficient (-1 to 1).
    """
    if len(rater1) != len(rater2) or not rater1:
        return 0.0

    n = len(rater1)
    all_categories = sorted(set(rater1) | set(rater2))

    # Build confusion matrix
    matrix = {}
    for cat in all_categories:
        matrix[cat] = {c: 0 for c in all_categories}
    for r1, r2 in zip(rater1, rater2):
        matrix[r1][r2] += 1

    # Observed agreement
    p_o = sum(matrix[c][c] for c in all_categories) / n

    # Expected agreement (by chance)
    p_e = 0.0
    for cat in all_categories:
        row_total = sum(matrix[cat].values()) / n
        col_total = sum(matrix[r][cat] for r in all_categories) / n
        p_e += row_total * col_total

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha(
    ratings_matrix: list[list[int | None]],
    level: str = "ordinal",
) -> float:
    """Compute Krippendorff's alpha for multiple raters.

    Args:
        ratings_matrix: List of lists, where each inner list contains
            scores from one rater across all items. None = missing.
        level: Measurement level ("nominal" or "ordinal").

    Returns:
        Krippendorff's alpha (-1 to 1). > 0.67 tentative, > 0.80 firm.
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return 0.0

    n_raters = len(ratings_matrix)
    n_items = len(ratings_matrix[0])

    # Collect all valid values
    all_values = sorted(set(
        v for rater in ratings_matrix for v in rater if v is not None
    ))
    if len(all_values) < 2:
        return 1.0

    # Build reliability data matrix
    # For each item, count how many raters assigned each value
    item_counts = []
    for item_idx in range(n_items):
        values = [ratings_matrix[r][item_idx] for r in range(n_raters)
                  if ratings_matrix[r][item_idx] is not None]
        if len(values) < 2:
            continue
        counts = Counter(values)
        item_counts.append((len(values), counts))

    if not item_counts:
        return 0.0

    # Observed disagreement
    d_o = 0.0
    total_pairs = 0
    for m_u, counts in item_counts:
        if m_u < 2:
            continue
        pairs_in_unit = m_u * (m_u - 1)
        total_pairs += pairs_in_unit
        for c, k in all_values, None:
            pass

    # Simplified computation using coincidence matrix
    coincidence = {v1: {v2: 0.0 for v2 in all_values} for v1 in all_values}
    n_total = 0

    for m_u, counts in item_counts:
        if m_u < 2:
            continue
        for v1 in all_values:
            for v2 in all_values:
                if v1 == v2:
                    n_ck = counts.get(v1, 0)
                    coincidence[v1][v2] += n_ck * (n_ck - 1) / (m_u - 1)
                else:
                    n_c = counts.get(v1, 0)
                    n_k = counts.get(v2, 0)
                    coincidence[v1][v2] += n_c * n_k / (m_u - 1)
        n_total += m_u

    if n_total < 2:
        return 0.0

    # Marginals
    n_c = {v: sum(coincidence[v].values()) for v in all_values}
    n_all = sum(n_c.values())

    if n_all == 0:
        return 0.0

    # Observed disagreement
    d_o = 0.0
    for v1 in all_values:
        for v2 in all_values:
            if v1 != v2:
                if level == "ordinal":
                    diff = (all_values.index(v1) - all_values.index(v2)) ** 2
                else:
                    diff = 1.0
                d_o += coincidence[v1][v2] * diff

    # Expected disagreement
    d_e = 0.0
    for v1 in all_values:
        for v2 in all_values:
            if v1 != v2:
                if level == "ordinal":
                    diff = (all_values.index(v1) - all_values.index(v2)) ** 2
                else:
                    diff = 1.0
                d_e += n_c.get(v1, 0) * n_c.get(v2, 0) * diff

    if d_e == 0:
        return 1.0

    d_e /= (n_all * (n_all - 1))
    d_o /= n_all

    return 1.0 - d_o / d_e if d_e > 0 else 1.0


def compute_inter_rater_reliability(
    results: dict,
) -> dict:
    """Compute inter-rater reliability from benchmark results with individual_scores.

    Args:
        results: Layer 1 results dict with individual_scores per prompt.

    Returns:
        Dict with per-dimension kappa, overall alpha, and agreement rates.
    """
    reliability = {}

    for dimension, dim_results in results.items():
        # Extract individual judge scores per prompt
        all_judge_scores = []
        for prompt_result in dim_results:
            individual = prompt_result.get("individual_scores", [])
            if individual:
                all_judge_scores.append(individual)

        if not all_judge_scores:
            reliability[dimension] = {
                "n_items": 0,
                "pairwise_kappa": [],
                "mean_kappa": 0.0,
                "exact_agreement_rate": 0.0,
            }
            continue

        n_judges = len(all_judge_scores[0])
        n_items = len(all_judge_scores)

        # Compute pairwise Cohen's kappa
        kappas = []
        for j1, j2 in combinations(range(n_judges), 2):
            r1 = [all_judge_scores[i][j1] for i in range(n_items)
                  if j1 < len(all_judge_scores[i]) and j2 < len(all_judge_scores[i])]
            r2 = [all_judge_scores[i][j2] for i in range(n_items)
                  if j1 < len(all_judge_scores[i]) and j2 < len(all_judge_scores[i])]
            if r1 and r2:
                kappas.append(cohens_kappa(r1, r2))

        # Exact agreement rate
        exact_agree = sum(
            1 for scores in all_judge_scores if len(set(scores)) == 1
        )

        reliability[dimension] = {
            "n_items": n_items,
            "n_judges": n_judges,
            "pairwise_kappa": [round(k, 3) for k in kappas],
            "mean_kappa": round(float(np.mean(kappas)), 3) if kappas else 0.0,
            "exact_agreement_rate": round(exact_agree / n_items, 3) if n_items else 0.0,
        }

    return reliability
