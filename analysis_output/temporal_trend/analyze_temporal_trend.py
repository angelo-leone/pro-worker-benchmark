"""
Standalone analysis: has there been a temporal trend in baseline pro-worker
behavior across LLM releases? (i.e., are newer models more "automating" and
less "augmenting"?)

Restricted, fair-comparison protocol:
  - 6 dimensions present in BOTH the Feb-2026 (v1) and Apr-2026 (v2) data:
      cognitive_forcing, complementarity, contrastive_explanation,
      draft_annotation, skill_preservation, uncertainty_transparency
  - 15 prompt IDs (cf_01..cf_15 etc.) present in both datasets
  - Baseline variant only (no system prompt). The trend question is about
    what models do by default, not what we can prompt them into.
  - PWI-6 = weighted mean across the 6 dims, weights renormalized to 1,
    rescaled to 0..100. Comparable across old and new.

Caveats noted in REPORT.md:
  * 10 models, time range ~21 months — narrow for trend detection.
  * Model-mix confound (different families, sizes, quantization).
  * Judge change between v1 (single judge) and v2 (3-judge median).
  * Several release dates are estimates from naming/family conventions.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent

COMMON_DIMS = [
    "cognitive_forcing",
    "complementarity",
    "contrastive_explanation",
    "draft_annotation",
    "skill_preservation",
    "uncertainty_transparency",
]

# Production weights (config.yaml) restricted to the 6 common dims, renormalized.
RAW_WEIGHTS_6 = {
    "cognitive_forcing": 0.15,
    "complementarity": 0.12,
    "contrastive_explanation": 0.10,
    "skill_preservation": 0.10,
    "uncertainty_transparency": 0.10,
    "draft_annotation": 0.08,
}
_w_sum = sum(RAW_WEIGHTS_6.values())
WEIGHTS_6 = {k: v / _w_sum for k, v in RAW_WEIGHTS_6.items()}
MAX_SCORE = 3.0

# Common prompt-ID prefixes per dim (cf_, com_, ce_, da_, sp_, ut_).
# Restrict to IDs 01..15 (present in both v1 and v2).
COMMON_PROMPT_IDS = {
    "cognitive_forcing": [f"cf_{i:02d}" for i in range(1, 16)],
    "complementarity": [f"co_{i:02d}" for i in range(1, 16)],
    "contrastive_explanation": [f"ce_{i:02d}" for i in range(1, 16)],
    "draft_annotation": [f"da_{i:02d}" for i in range(1, 16)],
    "skill_preservation": [f"sp_{i:02d}" for i in range(1, 16)],
    "uncertainty_transparency": [f"ut_{i:02d}" for i in range(1, 16)],
}

# Release dates. "verified" = public release date is well-known; "estimated"
# = inferred from family naming (no explicit date in the model id) and may
# be off by a quarter or so. Devstral has the explicit "2512" tag = Dec 2025.
MODEL_META: dict[str, dict] = {
    "Llama 3.1 70B":          {"date": date(2024, 7, 23),  "source": "verified",  "family": "Meta"},
    "Qwen2.5 72B":            {"date": date(2024, 9, 19),  "source": "verified",  "family": "Alibaba"},
    "Mistral Small 3.1":      {"date": date(2025, 3, 17),  "source": "verified",  "family": "Mistral"},
    "GPT-oss 120B":           {"date": date(2025, 8, 5),   "source": "verified",  "family": "OpenAI"},
    "Gemma 4 31B":            {"date": date(2025, 10, 1),  "source": "estimated", "family": "Google"},
    "Nemotron-Cascade 30B":   {"date": date(2025, 11, 1),  "source": "estimated", "family": "NVIDIA"},
    "DeepSeek V3.2":          {"date": date(2025, 11, 15), "source": "estimated", "family": "DeepSeek"},
    "Devstral-2 123B":        {"date": date(2025, 12, 1),  "source": "verified",  "family": "Mistral"},
    "Qwen3.5 27B":            {"date": date(2026, 1, 15),  "source": "estimated", "family": "Alibaba"},
    "GLM 5.1":                {"date": date(2026, 2, 1),   "source": "estimated", "family": "Zhipu"},
}

# Map result-file model_name strings to canonical names above.
NAME_REMAP = {
    "Llama 3.1 70B": "Llama 3.1 70B",
    "Qwen 2.5 72B": "Qwen2.5 72B",
    "Mistral Small 3.1 24B": "Mistral Small 3.1",
}


def canonical_name(raw: str) -> str:
    return NAME_REMAP.get(raw, raw)


def parse_timestamp(fname: str) -> str | None:
    m = re.search(r"_(\d{8})_(\d{6})\.json$", fname)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def load_baseline_runs() -> dict[str, list[dict]]:
    """Group baseline result files by canonical model name."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*_baseline_*.json")):
        with open(path) as f:
            data = json.load(f)
        if data.get("variant") != "baseline":
            continue
        name = canonical_name(data.get("model_name", ""))
        if name not in MODEL_META:
            continue
        by_model[name].append(data)
    return dict(by_model)


def collect_scores(runs: list[dict]) -> dict[str, list[float]]:
    """For one model, pool all valid scores per common dimension across files,
    restricted to the 15 shared prompt IDs.
    """
    out: dict[str, list[float]] = {d: [] for d in COMMON_DIMS}
    for run in runs:
        layer1 = run.get("layer1") or {}
        for dim in COMMON_DIMS:
            entries = layer1.get(dim) or []
            allowed = set(COMMON_PROMPT_IDS[dim])
            for e in entries:
                pid = e.get("prompt_id")
                if pid not in allowed:
                    continue
                # v2 has mean_score (multi-run aggregate); v1 has single score
                if "mean_score" in e and e["mean_score"] is not None:
                    s = e["mean_score"]
                else:
                    s = e.get("score")
                if s is None or s < 0:
                    continue
                out[dim].append(float(s))
    return out


def compute_pwi6(dim_means: dict[str, float]) -> float:
    """Weighted mean over the 6 common dimensions, scaled to 0..100."""
    return 100.0 * sum(WEIGHTS_6[d] * dim_means[d] for d in COMMON_DIMS) / MAX_SCORE


def main() -> int:
    by_model = load_baseline_runs()
    missing = sorted(set(MODEL_META) - set(by_model))
    if missing:
        print(f"warning: no baseline result files for: {missing}")

    rows = []
    raw_dim_scores = {}  # model -> dim -> list of scores (for stat tests)
    for model, runs in by_model.items():
        scores = collect_scores(runs)
        n_per_dim = {d: len(scores[d]) for d in COMMON_DIMS}
        if min(n_per_dim.values()) == 0:
            print(f"skipping {model}: missing scores for some common dim ({n_per_dim})")
            continue
        dim_means = {d: float(np.mean(scores[d])) for d in COMMON_DIMS}
        pwi6 = compute_pwi6(dim_means)
        meta = MODEL_META[model]
        row = {
            "model": model,
            "family": meta["family"],
            "release_date": meta["date"].isoformat(),
            "release_date_source": meta["source"],
            "PWI_6": round(pwi6, 2),
            "n_total": sum(n_per_dim.values()),
        }
        for d in COMMON_DIMS:
            row[f"mean_{d}"] = round(dim_means[d], 3)
        for d in COMMON_DIMS:
            row[f"n_{d}"] = n_per_dim[d]
        rows.append(row)
        raw_dim_scores[model] = scores

    df = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    df.to_csv(OUT_DIR / "pwi6_by_model.csv", index=False)
    print(f"wrote {OUT_DIR / 'pwi6_by_model.csv'} ({len(df)} models)")

    # Months since 2024-06-01 (smallest unit that keeps regression interpretable).
    epoch = date(2024, 6, 1)
    df["months_since_epoch"] = df["release_date"].apply(
        lambda s: (date.fromisoformat(s) - epoch).days / 30.4375
    )

    # Linear regression: PWI_6 ~ months
    x = df["months_since_epoch"].values
    y = df["PWI_6"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    n = len(df)
    df_dof = n - 2
    t_crit = stats.t.ppf(0.975, df_dof)
    slope_ci = (slope - t_crit * se, slope + t_crit * se)
    print()
    print("=== PWI-6 (baseline) vs months-since-2024-06 ===")
    print(f"  n models       : {n}")
    print(f"  slope          : {slope:+.3f} PWI-points / month")
    print(f"  95% CI         : [{slope_ci[0]:+.3f}, {slope_ci[1]:+.3f}]")
    print(f"  Pearson r      : {r:+.3f}")
    print(f"  p (two-sided)  : {p:.4f}")

    # Per-dimension regressions (mean dim score in 0..3 vs months)
    per_dim_stats = []
    for d in COMMON_DIMS:
        yy = df[f"mean_{d}"].values
        s, b, rr, pp, ss = stats.linregress(x, yy)
        per_dim_stats.append({
            "dimension": d,
            "slope_per_month": round(s, 4),
            "r": round(rr, 3),
            "p": round(pp, 4),
            "intercept": round(b, 3),
        })
    pd.DataFrame(per_dim_stats).to_csv(OUT_DIR / "per_dim_trend.csv", index=False)
    print()
    print("=== per-dimension slope (raw 0..3 score per month) ===")
    for r_ in per_dim_stats:
        print(f"  {r_['dimension']:28s}  slope={r_['slope_per_month']:+.4f}  r={r_['r']:+.3f}  p={r_['p']:.4f}")

    # ---------- figures ----------
    family_palette = {
        "Meta": "#1f77b4", "Alibaba": "#ff7f0e", "Mistral": "#2ca02c",
        "OpenAI": "#d62728", "Google": "#9467bd", "NVIDIA": "#76b900",
        "DeepSeek": "#17becf", "Zhipu": "#e377c2",
    }

    # Figure 1: PWI-6 vs release date
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dates_dt = pd.to_datetime(df["release_date"])
    for fam, sub in df.groupby("family"):
        sd = pd.to_datetime(sub["release_date"])
        ax.scatter(sd, sub["PWI_6"], s=110, color=family_palette.get(fam, "gray"),
                   edgecolor="black", linewidth=0.8, label=fam, zorder=3)
    for _, r_ in df.iterrows():
        marker = "" if r_["release_date_source"] == "verified" else " *"
        ax.annotate(r_["model"] + marker,
                    (pd.to_datetime(r_["release_date"]), r_["PWI_6"]),
                    xytext=(7, 4), textcoords="offset points",
                    fontsize=8, color="#222")
    # regression line over the date range
    xs = np.linspace(x.min(), x.max(), 100)
    ys = intercept + slope * xs
    line_dates = [epoch + pd.Timedelta(days=int(m * 30.4375)) for m in xs]
    ax.plot(line_dates, ys, color="black", linestyle="--", linewidth=1.4, alpha=0.7,
            label=f"OLS slope = {slope:+.2f} / month (p={p:.3f})")
    # 95% CI band on prediction
    yhat_lo = intercept + slope_ci[0] * xs
    yhat_hi = intercept + slope_ci[1] * xs
    ax.fill_between(line_dates,
                    np.minimum(yhat_lo, yhat_hi), np.maximum(yhat_lo, yhat_hi),
                    color="gray", alpha=0.12, label="95% CI on slope")
    ax.set_xlabel("Release date")
    ax.set_ylabel("PWI-6 (baseline, 0–100)")
    ax.set_title("Baseline pro-worker behavior across model releases\n"
                 "(restricted to 6 dimensions × 15 prompts shared by v1/v2 data)",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.text(0.01, 0.01, "* = release date estimated from family naming",
             fontsize=7, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_pwi6_vs_date.png", dpi=200)
    fig.savefig(OUT_DIR / "fig_pwi6_vs_date.pdf")
    plt.close(fig)

    # Figure 2: per-dimension trend (small multiples, 2x3)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, d in zip(axes.flatten(), COMMON_DIMS):
        yy = df[f"mean_{d}"].values
        for _, r_ in df.iterrows():
            ax.scatter(pd.to_datetime(r_["release_date"]), r_[f"mean_{d}"],
                       s=55, color=family_palette.get(r_["family"], "gray"),
                       edgecolor="black", linewidth=0.5, zorder=3)
        s, b, rr, pp, _ = stats.linregress(x, yy)
        ys_d = b + s * xs
        line_dates = [epoch + pd.Timedelta(days=int(m * 30.4375)) for m in xs]
        ax.plot(line_dates, ys_d, color="black", linestyle="--", linewidth=1.1, alpha=0.6)
        sig = "*" if pp < 0.05 else ""
        ax.set_title(f"{d}\nslope={s:+.3f}/mo, r={rr:+.2f}, p={pp:.3f}{sig}", fontsize=9)
        ax.set_ylim(0, 3)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("mean score (0–3)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Release date", fontsize=9)
    fig.suptitle("Per-dimension baseline trend across releases", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "fig_per_dim_trend.png", dpi=200)
    fig.savefig(OUT_DIR / "fig_per_dim_trend.pdf")
    plt.close(fig)

    # Figure 3: bucketed half-year comparison (more robust than exact dates)
    def bucket(d: str) -> str:
        dd = date.fromisoformat(d)
        half = "H1" if dd.month <= 6 else "H2"
        return f"{dd.year} {half}"
    df["bucket"] = df["release_date"].apply(bucket)
    bucket_order = sorted(df["bucket"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bucket_means, bucket_n = [], []
    for b in bucket_order:
        sub = df[df["bucket"] == b]
        bucket_means.append(sub["PWI_6"].mean())
        bucket_n.append(len(sub))
        for _, r_ in sub.iterrows():
            ax.scatter(b, r_["PWI_6"], s=80,
                       color=family_palette.get(r_["family"], "gray"),
                       edgecolor="black", linewidth=0.6, zorder=3)
    ax.plot(bucket_order, bucket_means, "k--", linewidth=1.4,
            label="bucket mean", zorder=2)
    for i, (b, m, nn) in enumerate(zip(bucket_order, bucket_means, bucket_n)):
        ax.annotate(f"n={nn}\nmean={m:.1f}", (i, m),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color="#222")
    ax.set_ylabel("PWI-6 (baseline)")
    ax.set_xlabel("Release half-year")
    ax.set_title("Baseline PWI-6 by half-year cohort", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_bucket.png", dpi=200)
    plt.close(fig)

    # Save a long-form per-dim score table for transparency
    long_rows = []
    for model, runs in by_model.items():
        if model not in df["model"].values:
            continue
        scores = raw_dim_scores[model]
        for dim, vals in scores.items():
            for v in vals:
                long_rows.append({"model": model, "dimension": dim, "score": v})
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "all_scores_long.csv", index=False)

    # ---------- stratified analysis: confound check ----------
    # The v1 (Feb 2026) data used a single judge; v2 (Apr 2026) used a
    # 3-judge median panel. If the apparent trend is driven by the protocol
    # break, it should disappear within either cohort.
    df["protocol"] = df["release_date"].apply(
        lambda s: "v1_single_judge" if date.fromisoformat(s) < date(2025, 6, 1)
        else "v2_panel_median"
    )
    print()
    print("=== stratified by judging protocol ===")
    for proto, sub in df.groupby("protocol"):
        if len(sub) < 3:
            print(f"  {proto}: n={len(sub)} — too few for regression "
                  f"(mean PWI-6 = {sub['PWI_6'].mean():.1f})")
            continue
        xs_ = sub["months_since_epoch"].values
        ys_ = sub["PWI_6"].values
        s_, b_, r_, p_, _ = stats.linregress(xs_, ys_)
        print(f"  {proto}: n={len(sub)}  slope={s_:+.3f}/mo  r={r_:+.3f}  p={p_:.4f}  "
              f"mean PWI-6={ys_.mean():.1f}")
    # Welch t-test between cohorts (testing if the protocols differ in level)
    grp1 = df[df["protocol"] == "v1_single_judge"]["PWI_6"].values
    grp2 = df[df["protocol"] == "v2_panel_median"]["PWI_6"].values
    if len(grp1) >= 2 and len(grp2) >= 2:
        t, p_t = stats.ttest_ind(grp1, grp2, equal_var=False)
        print(f"  Welch t-test (v1 vs v2 mean): t={t:+.2f}, p={p_t:.4f}, "
              f"v1 mean={grp1.mean():.1f}, v2 mean={grp2.mean():.1f}")
    df.to_csv(OUT_DIR / "pwi6_by_model.csv", index=False)  # rewrite with protocol col

    # Figure 4: stratified scatter showing the protocol break
    fig, ax = plt.subplots(figsize=(9, 5))
    for proto, marker, label in [("v1_single_judge", "s", "v1 (single-judge, Feb 2026 data)"),
                                  ("v2_panel_median", "o", "v2 (3-judge median, Apr 2026 data)")]:
        sub = df[df["protocol"] == proto]
        ax.scatter(pd.to_datetime(sub["release_date"]), sub["PWI_6"],
                   s=110, marker=marker, edgecolor="black", linewidth=0.8,
                   label=label,
                   color="#d62728" if proto == "v1_single_judge" else "#1f77b4",
                   zorder=3)
        for _, r_ in sub.iterrows():
            ax.annotate(r_["model"],
                        (pd.to_datetime(r_["release_date"]), r_["PWI_6"]),
                        xytext=(7, 4), textcoords="offset points",
                        fontsize=8, color="#222")
    # Per-cohort regression lines
    for proto, color in [("v1_single_judge", "#d62728"), ("v2_panel_median", "#1f77b4")]:
        sub = df[df["protocol"] == proto]
        if len(sub) < 3:
            continue
        xs_ = sub["months_since_epoch"].values
        ys_ = sub["PWI_6"].values
        s_, b_, r_, p_, _ = stats.linregress(xs_, ys_)
        xs_pl = np.linspace(xs_.min(), xs_.max(), 50)
        ys_pl = b_ + s_ * xs_pl
        line_dates = [epoch + pd.Timedelta(days=int(m * 30.4375)) for m in xs_pl]
        ax.plot(line_dates, ys_pl, "--", color=color, linewidth=1.4, alpha=0.6,
                label=f"  ↪ within {proto}: slope={s_:+.2f}/mo (p={p_:.3f})")
    ax.set_xlabel("Release date")
    ax.set_ylabel("PWI-6 (baseline, 0–100)")
    ax.set_title("Apparent 'temporal trend' is largely a judging-protocol break\n"
                 "(v1 used a single judge; v2 uses 3-judge median)",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_protocol_confound.png", dpi=200)
    fig.savefig(OUT_DIR / "fig_protocol_confound.pdf")
    plt.close(fig)

    print()
    print(f"figures + CSVs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
