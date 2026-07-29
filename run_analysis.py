"""
Full analysis pipeline for the Pro-Worker AI Benchmark.
Computes all statistics needed for the NeurIPS 2026 paper:
- PWI scores with Bayesian confidence intervals
- Per-dimension breakdowns (baseline vs prompted)
- System prompt effect sizes (Cohen's d)
- Inter-rater reliability (if multi-judge data available)
- Construct validity (inter-dimension correlations, Cronbach's alpha)
- Weight sensitivity analysis

Usage:
    python3 run_analysis.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.statistics import (
    bayesian_score_estimate,
    compute_effect_size,
    bootstrap_pwi_ci,
)

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Canonical PWI weights, loaded from config.yaml (the single source of truth).
# Set PWB_WEIGHT_SCHEME=legacy_weights_v2_0 to reproduce the published v2.0.0 numbers.
from src.analysis import load_weights  # noqa: E402

WEIGHTS = load_weights(os.environ.get("PWB_WEIGHT_SCHEME", "weights"))

# Reporting and weighting are separate concerns. Every Layer-1 dimension is reported, including
# those excluded from the composite by the reliability cap, because they remain informative as
# diagnostics. Only WEIGHTS decides what enters PWI.
N_JUDGES = 3
MAX_SCORE = 3


def _qwk(a, b, k=MAX_SCORE + 1):
    """Quadratic-weighted Cohen's kappa on a 0..k-1 ordinal scale."""
    a, b = np.asarray(a), np.asarray(b)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    E = np.outer(np.bincount(a, minlength=k), np.bincount(b, minlength=k)) / len(a)
    i, j = np.mgrid[0:k, 0:k]
    W = (i - j) ** 2 / (k - 1) ** 2
    den = (W * E).sum()
    return 1.0 - (W * O).sum() / den if den > 0 else float("nan")


def _krippendorff_interval(units, k=MAX_SCORE + 1):
    """Krippendorff's alpha with an interval difference function."""
    num, pairs, values = 0.0, 0, []
    for u in units:
        for x in u:
            for y in u:
                if x is not y:
                    num += (x - y) ** 2
        pairs += len(u) * (len(u) - 1)
        values.extend(u)
    Do = num / pairs
    v = np.asarray(values)
    counts = np.bincount(v, minlength=k).astype(float)
    i, j = np.mgrid[0:k, 0:k]
    De = ((counts[:, None] * counts[None, :]) * (i - j) ** 2).sum() / (len(v) * (len(v) - 1))
    return 1.0 - Do / De if De > 0 else float("nan")


def collect_judge_triples(results: dict) -> tuple[dict, int]:
    """Group complete three-judge score sets by dimension.

    Only complete triples are counted. Including runs where a judge failed schema validation
    would mix two-rater and three-rater agreement, and two raters agree exactly far more often
    than three, which inflates the statistic. Returns the groups and the excluded count.
    """
    triples, n_incomplete = defaultdict(list), 0
    for data in results.values():
        for dim, prompts in data.get("layer1", {}).items():
            for p in prompts:
                for r in (p["runs"] if "runs" in p else [p]):
                    s = r.get("individual_scores")
                    if not s:
                        continue
                    if len(s) != N_JUDGES or any(
                        x is None or x < 0 or x > MAX_SCORE for x in s
                    ):
                        n_incomplete += 1
                        continue
                    triples[dim].append([int(x) for x in s])
    return triples, n_incomplete


def irr_metrics(triples: list) -> dict:
    """Reliability metrics for a list of complete three-judge score triples."""
    arr = np.array(triples)
    ks = [_qwk(arr[:, x], arr[:, y]) for x, y in ((0, 1), (0, 2), (1, 2))]
    return {
        "n": len(triples),
        "exact": float((arr.max(1) == arr.min(1)).mean() * 100),
        "adjacent": float(((arr.max(1) - arr.min(1)) <= 1).mean() * 100),
        "qwk": float(np.nanmean(ks)),
        "alpha": float(_krippendorff_interval(triples)),
        "spread": float((arr.max(1) - arr.min(1)).mean()),
    }


DIMENSIONS = [
    "cognitive_forcing",
    "contrastive_explanation",
    "skill_preservation",
    "draft_annotation",
    "uncertainty_transparency",
    "complementarity",
    "anti_sycophancy",
    "metacognitive_calibration",
    "appropriate_reliance",
    "ethical_surfacing",
]

# Only use v2 results (11 dimensions, multi-run, multi-judge)
V2_MODELS = [
    "DeepSeek V3.2",
    "Devstral-2 123B",
    "Gemma 4 31B",
    "GPT-oss 120B",
    "Nemotron-Cascade 30B",
    "Qwen3.5 27B",
    "GLM 5.1",
]


def load_latest_results() -> dict:
    """Load the latest result file for each model+variant."""
    latest = {}
    for f in sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime):
        data = json.load(open(f))
        name = data.get("model_name", "?")
        variant = data.get("variant", "?")

        # Only v2 models
        if name not in V2_MODELS:
            continue

        key = f"{name}|{variant}"
        # Check it has all 3 layers
        if "layer1" in data and "layer2" in data and "layer3" in data:
            latest[key] = data

    return latest


def extract_l1_scores(data: dict) -> dict[str, list[float]]:
    """Extract per-dimension score lists from Layer 1 data."""
    dim_scores = {}
    for dim, prompts in data.get("layer1", {}).items():
        scores = []
        for p in prompts:
            if "runs" in p:
                for r in p["runs"]:
                    if r.get("score", -1) >= 0:
                        scores.append(float(r["score"]))
            elif p.get("score", -1) >= 0:
                scores.append(float(p["score"]))
        if scores:
            dim_scores[dim] = scores
    return dim_scores


def extract_l1_prompt_means(data: dict) -> dict[str, list[float]]:
    """Extract per-prompt mean scores (across runs) for each dimension."""
    dim_means = {}
    for dim, prompts in data.get("layer1", {}).items():
        means = []
        for p in prompts:
            if "runs" in p:
                valid = [r["score"] for r in p["runs"] if r.get("score", -1) >= 0]
                if valid:
                    means.append(float(np.mean(valid)))
            elif p.get("score", -1) >= 0:
                means.append(float(p["score"]))
        if means:
            dim_means[dim] = means
    return dim_means


def compute_pwi(dim_means: dict[str, float], weights: dict = WEIGHTS) -> float:
    """Compute PWI from dimension means."""
    total_w = sum(weights.get(d, 0) for d in dim_means)
    if total_w == 0:
        return 0.0
    weighted = sum(dim_means.get(d, 0) / 3.0 * weights.get(d, 0) for d in dim_means)
    return (weighted / total_w) * 100


def run_analysis():
    print("=" * 70)
    print("PRO-WORKER AI BENCHMARK — FULL ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading results...")
    results = load_latest_results()
    print(f"  Loaded {len(results)} model-variant pairs")

    models = sorted(set(k.split("|")[0] for k in results))
    print(f"  Models: {', '.join(models)}")

    triples, n_incomplete = collect_judge_triples(results)

    # =========================================================
    # 1. PWI SCORES WITH BAYESIAN CIs
    # =========================================================
    print("\n" + "=" * 70)
    print("1. PRO-WORKER INDEX (PWI) SCORES")
    print("=" * 70)

    pwi_table = []
    all_dim_scores = {}  # for later analysis

    for model in models:
        for variant in ["baseline", "with_system_prompt"]:
            key = f"{model}|{variant}"
            if key not in results:
                continue

            data = results[key]
            dim_scores = extract_l1_scores(data)
            dim_means_dict = {d: float(np.mean(s)) for d, s in dim_scores.items()}

            # Store for later
            all_dim_scores[key] = dim_scores

            # Compute PWI with bootstrap CI
            pwi_ci = bootstrap_pwi_ci(dim_scores, WEIGHTS)

            pwi_table.append({
                "Model": model,
                "Variant": variant,
                "PWI": round(pwi_ci["pwi_mean"], 1),
                "CI_lower": round(pwi_ci["ci_lower"], 1),
                "CI_upper": round(pwi_ci["ci_upper"], 1),
            })

    pwi_df = pd.DataFrame(pwi_table)
    print("\n" + pwi_df.to_string(index=False))

    # Save
    pwi_df.to_csv(OUTPUT_DIR / "pwi_scores.csv", index=False)

    # =========================================================
    # 2. PER-DIMENSION BREAKDOWN
    # =========================================================
    print("\n" + "=" * 70)
    print("2. PER-DIMENSION SCORES (Mean ± Bayesian 95% HDI)")
    print("=" * 70)

    dim_table = []
    for model in models:
        for variant in ["baseline", "with_system_prompt"]:
            key = f"{model}|{variant}"
            if key not in all_dim_scores:
                continue
            for dim in DIMENSIONS:
                scores = all_dim_scores[key].get(dim, [])
                if not scores:
                    continue
                bayes = bayesian_score_estimate(scores)
                dim_table.append({
                    "Model": model,
                    "Variant": variant,
                    "Dimension": dim,
                    "Mean": round(bayes["posterior_mean"], 2),
                    "HDI_lower": round(bayes["hdi_lower"], 2),
                    "HDI_upper": round(bayes["hdi_upper"], 2),
                    "N": bayes["n_observations"],
                })

    dim_df = pd.DataFrame(dim_table)
    dim_df.to_csv(OUTPUT_DIR / "dimension_scores.csv", index=False)

    # Print summary: baseline vs prompted per dimension (averaged across models)
    print("\nDimension averages across all models:")
    print(f"{'Dimension':35s} {'Baseline':>10s} {'Prompted':>10s} {'Delta':>8s}")
    print("-" * 65)
    for dim in DIMENSIONS:
        baseline_scores = []
        prompted_scores = []
        for model in models:
            b_key = f"{model}|baseline"
            p_key = f"{model}|with_system_prompt"
            if b_key in all_dim_scores and dim in all_dim_scores[b_key]:
                baseline_scores.extend(all_dim_scores[b_key][dim])
            if p_key in all_dim_scores and dim in all_dim_scores[p_key]:
                prompted_scores.extend(all_dim_scores[p_key][dim])
        b_mean = np.mean(baseline_scores) if baseline_scores else 0
        p_mean = np.mean(prompted_scores) if prompted_scores else 0
        delta = p_mean - b_mean
        print(f"  {dim:33s} {b_mean:8.2f}/3  {p_mean:8.2f}/3  {delta:+6.2f}")

    # One diagnostic row per dimension: effect, its uncertainty, how reliably the dimension is
    # measured, and how consistent the direction is across models. Reviewers asked for effect
    # size and reliability to be legible side by side rather than in separate appendices.
    rng = np.random.default_rng(0)
    diag_rows = []
    for dim in DIMENSIONS:
        b_all, p_all, improving = [], [], 0
        for model in models:
            b = all_dim_scores.get(f"{model}|baseline", {}).get(dim, [])
            p = all_dim_scores.get(f"{model}|with_system_prompt", {}).get(dim, [])
            if not b or not p:
                continue
            b_all.extend(b)
            p_all.extend(p)
            improving += int(np.mean(p) > np.mean(b))
        if not b_all or not p_all:
            continue
        b_arr, p_arr = np.asarray(b_all), np.asarray(p_all)
        boots = [
            rng.choice(p_arr, p_arr.size).mean() - rng.choice(b_arr, b_arr.size).mean()
            for _ in range(2000)
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        m = irr_metrics(triples[dim]) if triples.get(dim) else None
        diag_rows.append({
            "Dimension": dim,
            "Baseline": b_arr.mean(),
            "Prompted": p_arr.mean(),
            "Delta": p_arr.mean() - b_arr.mean(),
            "CI_lower": lo,
            "CI_upper": hi,
            "QWK": m["qwk"] if m else float("nan"),
            "Spread": m["spread"] if m else float("nan"),
            "Models_improving": improving,
            "Weight": WEIGHTS.get(dim, 0.0),
        })
    pd.DataFrame(diag_rows).to_csv(OUTPUT_DIR / "dimension_diagnostics.csv", index=False)

    # =========================================================
    # 3. SYSTEM PROMPT EFFECT SIZES (Cohen's d)
    # =========================================================
    print("\n" + "=" * 70)
    print("3. SYSTEM PROMPT EFFECT SIZES (Cohen's d)")
    print("=" * 70)

    effect_table = []
    for model in models:
        b_key = f"{model}|baseline"
        p_key = f"{model}|with_system_prompt"
        if b_key not in all_dim_scores or p_key not in all_dim_scores:
            continue

        for dim in DIMENSIONS:
            b_scores = all_dim_scores[b_key].get(dim, [])
            p_scores = all_dim_scores[p_key].get(dim, [])
            if not b_scores or not p_scores:
                continue
            effect = compute_effect_size(b_scores, p_scores)
            effect_table.append({
                "Model": model,
                "Dimension": dim,
                "Cohens_d": round(effect["cohens_d"], 3),
                "Interpretation": effect["interpretation"],
                "Baseline_mean": round(effect["baseline_mean"], 2),
                "Prompted_mean": round(effect["prompted_mean"], 2),
            })

    effect_df = pd.DataFrame(effect_table)
    effect_df.to_csv(OUTPUT_DIR / "effect_sizes.csv", index=False)

    # Print summary heatmap
    print(f"\n{'Model':25s}", end="")
    for dim in DIMENSIONS:
        print(f" {dim[:6]:>7s}", end="")
    print("   PWI_d")
    print("-" * (25 + 8 * len(DIMENSIONS) + 8))

    for model in models:
        print(f"  {model:23s}", end="")
        b_key = f"{model}|baseline"
        p_key = f"{model}|with_system_prompt"
        for dim in DIMENSIONS:
            row = effect_df[(effect_df["Model"] == model) & (effect_df["Dimension"] == dim)]
            if not row.empty:
                d = row.iloc[0]["Cohens_d"]
                print(f" {d:+7.2f}", end="")
            else:
                print(f"     N/A", end="")

        # Overall PWI effect
        b_scores_all = []
        p_scores_all = []
        for dim in DIMENSIONS:
            b_scores_all.extend(all_dim_scores.get(b_key, {}).get(dim, []))
            p_scores_all.extend(all_dim_scores.get(p_key, {}).get(dim, []))
        if b_scores_all and p_scores_all:
            overall = compute_effect_size(b_scores_all, p_scores_all)
            print(f"  {overall['cohens_d']:+6.2f}")
        else:
            print("    N/A")

    # =========================================================
    # 4. INTER-RATER RELIABILITY
    # =========================================================
    print("\n" + "=" * 70)
    print("4. INTER-RATER RELIABILITY")
    print("=" * 70)

    # Check if we have individual judge scores
    has_irr = False
    for key, data in results.items():
        for dim, prompts in data.get("layer1", {}).items():
            for p in prompts:
                if "runs" in p:
                    for r in p["runs"]:
                        if r.get("individual_scores"):
                            has_irr = True
                            break
                elif p.get("individual_scores"):
                    has_irr = True
                if has_irr:
                    break
            if has_irr:
                break
        if has_irr:
            break

    if has_irr:
        irr_rows = []
        print(
            f"\n{'Dimension':30s} {'N':>6s} {'Exact':>7s} {'Adj':>7s} "
            f"{'QW-kappa':>9s} {'alpha':>7s} {'Spread':>7s}"
        )
        print("-" * 80)
        for dim in DIMENSIONS + ["POOLED"]:
            t = (
                [x for d in DIMENSIONS for x in triples[d]]
                if dim == "POOLED"
                else triples.get(dim, [])
            )
            if not t:
                continue
            m = irr_metrics(t)
            # Full precision: the paper rounds these for display, and a consistency check
            # that compares against an already-rounded CSV double-rounds and false-alarms.
            irr_rows.append({"Dimension": dim, "N": m["n"], **{
                k: m[k] for k in ("exact", "adjacent", "qwk", "alpha", "spread")}})
            print(
                f"  {dim:28s} {m['n']:6d} {m['exact']:6.1f}% {m['adjacent']:6.1f}% "
                f"{m['qwk']:9.3f} {m['alpha']:7.3f} {m['spread']:7.2f}"
            )
        print(f"\n  Excluded {n_incomplete} runs lacking a complete three-judge score set.")
        pd.DataFrame(irr_rows).to_csv(OUTPUT_DIR / "irr_table.csv", index=False)
    else:
        print("  No individual judge scores found — single-judge data.")

    # =========================================================
    # 5. CONSTRUCT VALIDITY
    # =========================================================
    print("\n" + "=" * 70)
    print("5. CONSTRUCT VALIDITY")
    print("=" * 70)

    # Inter-dimension correlations (discriminant validity)
    print("\n5a. Inter-dimension correlations (discriminant validity)")
    print("    Target: all |r| < 0.70")

    # Aggregate all scores per dimension across all models
    agg_dim = {d: [] for d in DIMENSIONS}
    for key, dim_scores in all_dim_scores.items():
        for dim in DIMENSIONS:
            if dim in dim_scores:
                # Use prompt-level means for correlation
                prompt_means = extract_l1_prompt_means(results[key])
                if dim in prompt_means:
                    agg_dim[dim].extend(prompt_means[dim])

    # Align lengths
    min_len = min(len(v) for v in agg_dim.values() if v) if agg_dim else 0
    if min_len > 10:
        corr_matrix = np.zeros((len(DIMENSIONS), len(DIMENSIONS)))
        flagged = []
        for i, d1 in enumerate(DIMENSIONS):
            for j, d2 in enumerate(DIMENSIONS):
                v1 = np.array(agg_dim[d1][:min_len])
                v2 = np.array(agg_dim[d2][:min_len])
                if np.std(v1) > 0 and np.std(v2) > 0:
                    r = float(np.corrcoef(v1, v2)[0, 1])
                else:
                    r = 0.0
                corr_matrix[i, j] = r
                if i < j and abs(r) > 0.70:
                    flagged.append((d1, d2, r))

        # Print correlation matrix
        print(f"\n{'':15s}", end="")
        for d in DIMENSIONS:
            print(f" {d[:5]:>6s}", end="")
        print()
        for i, d1 in enumerate(DIMENSIONS):
            print(f"  {d1[:13]:13s}", end="")
            for j, d2 in enumerate(DIMENSIONS):
                r = corr_matrix[i, j]
                marker = "*" if abs(r) > 0.70 and i != j else " "
                print(f" {r:5.2f}{marker}", end="")
            print()

        if flagged:
            print(f"\n  WARNING: {len(flagged)} pairs exceed 0.70:")
            for d1, d2, r in flagged:
                print(f"    {d1} × {d2}: r={r:.3f}")
        else:
            print(f"\n  PASS: All inter-dimension correlations below 0.70")

        # Save correlation matrix
        corr_df = pd.DataFrame(corr_matrix, index=DIMENSIONS, columns=DIMENSIONS)
        corr_df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    else:
        print("  Insufficient data for correlation analysis")

    # =========================================================
    # 6. WEIGHT SENSITIVITY
    # =========================================================
    print("\n" + "=" * 70)
    print("6. WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    alt_weights = {
        "default": WEIGHTS,
        "equal": {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS},
        "cf_heavy": {**{d: 0.07 for d in DIMENSIONS}, "cognitive_forcing": 0.30},
        "anti_syc_heavy": {**{d: 0.07 for d in DIMENSIONS}, "anti_sycophancy": 0.30},
    }

    # Normalize weights
    for scheme_name, w in alt_weights.items():
        total = sum(w.values())
        alt_weights[scheme_name] = {d: v / total for d, v in w.items()}

    print(f"\n{'Model':25s}", end="")
    for scheme in alt_weights:
        print(f" {scheme[:10]:>11s}", end="")
    print("  Ranking stable?")
    print("-" * (25 + 12 * len(alt_weights) + 18))

    default_ranking = []
    scheme_rankings = {s: [] for s in alt_weights}

    for model in models:
        print(f"  {model:23s}", end="")
        b_key = f"{model}|with_system_prompt"
        if b_key not in all_dim_scores:
            b_key = f"{model}|baseline"

        dim_means = {}
        for dim in DIMENSIONS:
            scores = all_dim_scores.get(b_key, {}).get(dim, [])
            if scores:
                dim_means[dim] = float(np.mean(scores))

        for scheme_name, weights in alt_weights.items():
            pwi = compute_pwi(dim_means, weights)
            print(f" {pwi:10.1f}%", end="")
            scheme_rankings[scheme_name].append((model, pwi))

        print()

    # Check ranking stability
    default_order = [m for m, _ in sorted(scheme_rankings["default"], key=lambda x: -x[1])]
    print(f"\n  Default ranking: {' > '.join(default_order)}")
    for scheme_name, ranking in scheme_rankings.items():
        order = [m for m, _ in sorted(ranking, key=lambda x: -x[1])]
        stable = order == default_order
        if not stable:
            print(f"  {scheme_name}: {' > '.join(order)} {'(SAME)' if stable else '(CHANGED)'}")

    # =========================================================
    # 7. LAYER 2 & 3 SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("7. LAYER 2 (MULTI-TURN) & LAYER 3 (ADVERSARIAL) SUMMARY")
    print("=" * 70)

    for model in models:
        for variant in ["baseline", "with_system_prompt"]:
            key = f"{model}|{variant}"
            if key not in results:
                continue
            data = results[key]

            # Layer 2
            l2 = data.get("layer2", [])
            if l2:
                avg_scores = [s.get("average_score", 0) for s in l2 if s.get("average_score")]
                behavior_rates = [s.get("behavior_pass_rate", 0) for s in l2 if s.get("behavior_pass_rate") is not None]
                l2_avg = np.mean(avg_scores) if avg_scores else 0
                l2_behavior = np.mean(behavior_rates) if behavior_rates else 0
            else:
                l2_avg = l2_behavior = 0

            # Layer 3
            l3 = data.get("layer3", [])
            if l3:
                l3_scores = [s.get("score", -1) for s in l3 if s.get("score", -1) >= 0]
                l3_avg = np.mean(l3_scores) if l3_scores else 0
            else:
                l3_avg = 0

            print(f"  {model:25s} {variant:20s} L2={l2_avg:.2f}/3 (behavior={l2_behavior:.0%}) L3={l3_avg:.2f}/3")

    # Pooled per-layer means, for the Figure 2 summary panel. Layer 1 pools every run-level
    # score across dimensions; Layers 2 and 3 pool their native units.
    layer_rows = []
    for variant in ["baseline", "with_system_prompt"]:
        l1v, l2v, l3v = [], [], []
        for model in models:
            data = results.get(f"{model}|{variant}")
            if not data:
                continue
            for dim in DIMENSIONS:
                l1v.extend(all_dim_scores.get(f"{model}|{variant}", {}).get(dim, []))
            l2v.extend(s["average_score"] for s in data.get("layer2", [])
                       if s.get("average_score") is not None)
            l3v.extend(s["score"] for s in data.get("layer3", [])
                       if s.get("score", -1) >= 0)
        for layer, vals in (("Layer 1", l1v), ("Layer 2", l2v), ("Layer 3", l3v)):
            layer_rows.append({"Layer": layer, "Variant": variant,
                               "Mean": float(np.mean(vals)), "N": len(vals)})
    pd.DataFrame(layer_rows).to_csv(OUTPUT_DIR / "layer_summary.csv", index=False)

    # =========================================================
    # 8. SUMMARY TABLE FOR PAPER
    # =========================================================
    print("\n" + "=" * 70)
    print("8. PAPER-READY SUMMARY TABLE")
    print("=" * 70)

    summary = []
    for model in models:
        b_key = f"{model}|baseline"
        p_key = f"{model}|with_system_prompt"

        b_dims = {d: float(np.mean(s)) for d, s in all_dim_scores.get(b_key, {}).items()}
        p_dims = {d: float(np.mean(s)) for d, s in all_dim_scores.get(p_key, {}).items()}

        b_pwi = compute_pwi(b_dims)
        p_pwi = compute_pwi(p_dims)
        delta = p_pwi - b_pwi

        summary.append({
            "Model": model,
            "PWI_baseline": round(b_pwi, 1),
            "PWI_prompted": round(p_pwi, 1),
            "Delta_PWI": round(delta, 1),
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("PWI_prompted", ascending=False)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

    # =========================================================
    # SAVE ALL DATA
    # =========================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    run_analysis()
