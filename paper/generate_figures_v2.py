"""
Generate figures for the Pro-Worker AI Benchmark v2.0 paper.
Uses the analysis output CSVs.
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

OUTPUT_DIR = "figures"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paper-quality settings
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Model display names (short)
MODEL_SHORT = {
    "DeepSeek V3.2": "DeepSeek",
    "Devstral-2 123B": "Devstral",
    "GLM 5.1": "GLM",
    "GPT-oss 120B": "GPT-oss",
    "Gemma 4 31B": "Gemma",
    "Nemotron-Cascade 30B": "Nemotron",
    "Qwen3.5 27B": "Qwen3.5",
}

DIM_SHORT = {
    "cognitive_forcing": "Cog. Forcing",
    "contrastive_explanation": "Contrastive",
    "skill_preservation": "Skill Pres.",
    "draft_annotation": "Draft Annot.",
    "uncertainty_transparency": "Uncertainty",
    "complementarity": "Complement.",
    "anti_sycophancy": "Anti-Syc.",
    "metacognitive_calibration": "Meta-Calib.",
    "appropriate_reliance": "Approp. Rel.",
    "ethical_surfacing": "Ethical Surf.",
}

DIMENSIONS = list(DIM_SHORT.keys())


def fig1_pwi_comparison():
    """Figure 1: PWI scores — baseline vs prompted for all models."""
    df = pd.read_csv("/Users/angelo.leone/Documents/pro-worker-benchmark/analysis_output/pwi_scores.csv")

    models = sorted(df["Model"].unique(), key=lambda m: -df[df["Model"] == m]["PWI"].max())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    baseline = [df[(df["Model"] == m) & (df["Variant"] == "baseline")]["PWI"].values[0] for m in models]
    prompted = [df[(df["Model"] == m) & (df["Variant"] == "with_system_prompt")]["PWI"].values[0] for m in models]

    # CI error bars
    b_ci_low = [df[(df["Model"] == m) & (df["Variant"] == "baseline")]["CI_lower"].values[0] for m in models]
    b_ci_high = [df[(df["Model"] == m) & (df["Variant"] == "baseline")]["CI_upper"].values[0] for m in models]
    p_ci_low = [df[(df["Model"] == m) & (df["Variant"] == "with_system_prompt")]["CI_lower"].values[0] for m in models]
    p_ci_high = [df[(df["Model"] == m) & (df["Variant"] == "with_system_prompt")]["CI_upper"].values[0] for m in models]

    b_err = [[b - bl for b, bl in zip(baseline, b_ci_low)],
             [bh - b for b, bh in zip(baseline, b_ci_high)]]
    p_err = [[p - pl for p, pl in zip(prompted, p_ci_low)],
             [ph - p for p, ph in zip(prompted, p_ci_high)]]

    bars1 = ax.bar(x - width/2, baseline, width, label="Baseline", color="#90CAF9",
                   yerr=b_err, capsize=3, error_kw={"linewidth": 0.8})
    bars2 = ax.bar(x + width/2, prompted, width, label="With System Prompt", color="#1565C0",
                   yerr=p_err, capsize=3, error_kw={"linewidth": 0.8})

    # Add delta labels
    for i, (b, p) in enumerate(zip(baseline, prompted)):
        delta = p - b
        ax.annotate(f"+{delta:.0f}", (x[i] + width/2, p + 3),
                    ha="center", va="bottom", fontsize=7, color="#1565C0", fontweight="bold")

    ax.set_ylabel("Pro-Worker Index (PWI)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_title("Pro-Worker Index: Baseline vs. System Prompt")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_pwi_comparison.pdf")
    plt.savefig(f"{OUTPUT_DIR}/fig1_pwi_comparison.png")
    plt.close()
    print("  fig1_pwi_comparison.pdf")


def fig2_dimension_heatmap():
    """Figure 2: two panels. (a) per-layer baseline-vs-prompted summary; (b) a
    dimension heatmap clustered into reliable and low-reliability groups, baseline beside
    prompted, at legible type. Replaces the crowded single-panel prompted-only heatmap."""
    base = "/Users/angelo.leone/Documents/pro-worker-benchmark/analysis_output"
    layer = pd.read_csv(f"{base}/layer_summary.csv")
    diag = pd.read_csv(f"{base}/dimension_diagnostics.csv")

    # Dimensions clustered by reliability, most reliable first within each block.
    reliable = ["complementarity", "anti_sycophancy", "cognitive_forcing",
                "draft_annotation", "ethical_surfacing", "contrastive_explanation"]
    low = ["skill_preservation", "metacognitive_calibration",
           "uncertainty_transparency", "appropriate_reliance"]
    order = reliable + low
    diag = diag.set_index("Dimension")

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.35], wspace=0.28)

    # Panel (a): per-layer summary.
    axA = fig.add_subplot(gs[0])
    layers = ["Layer 1", "Layer 2", "Layer 3"]
    b = [layer[(layer.Layer == l) & (layer.Variant == "baseline")]["Mean"].iloc[0] for l in layers]
    p = [layer[(layer.Layer == l) & (layer.Variant == "with_system_prompt")]["Mean"].iloc[0] for l in layers]
    y = np.arange(len(layers))
    axA.barh(y + 0.2, b, 0.38, label="Baseline", color="#c2703d")
    axA.barh(y - 0.2, p, 0.38, label="Prompted", color="#3d7ac2")
    for yi, (bv, pv) in enumerate(zip(b, p)):
        axA.text(bv + 0.05, yi + 0.2, f"{bv:.2f}", va="center", fontsize=8)
        axA.text(pv + 0.05, yi - 0.2, f"{pv:.2f}", va="center", fontsize=8)
    axA.set_yticks(y)
    axA.set_yticklabels(["Layer 1\nsingle-turn", "Layer 2\nmulti-turn", "Layer 3\nadversarial"])
    axA.set_xlim(0, 3)
    axA.set_xlabel("Mean score (0–3)")
    axA.invert_yaxis()
    handles, labels_ = axA.get_legend_handles_labels()
    axA.set_title("(a) Effect by layer, pooled")
    axA.spines[["top", "right"]].set_visible(False)

    # Panel (b): clustered baseline|prompted heatmap.
    axB = fig.add_subplot(gs[1])
    matrix = np.array([[diag.loc[d, "Baseline"], diag.loc[d, "Prompted"]] for d in order])
    im = axB.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["Baseline", "Prompted"])
    axB.set_yticks(range(len(order)))
    labels = [f"{DIM_SHORT[d]}  (κ={diag.loc[d,'QWK']:.2f})" for d in order]
    axB.set_yticklabels(labels)
    for i, d in enumerate(order):
        for j in range(2):
            v = matrix[i, j]
            axB.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v < 1.0 or v > 2.5 else "black")
    # divider between reliable and low-reliability blocks
    axB.axhline(len(reliable) - 0.5, color="black", lw=1.5)
    axB.text(1.62, (len(reliable) - 1) / 2, "κ ≥ 0.70", rotation=90,
             va="center", ha="center", fontsize=8)
    axB.text(1.62, len(reliable) + (len(low) - 1) / 2, "κ < 0.70", rotation=90,
             va="center", ha="center", fontsize=8, color="#888")
    axB.set_title("(b) Per-dimension score, clustered by judge reliability")
    fig.colorbar(im, ax=axB, label="Score (0–3)", shrink=0.85, pad=0.12)

    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.22, -0.04))

    plt.savefig(f"{OUTPUT_DIR}/fig2_dimension_heatmap.pdf", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/fig2_dimension_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  fig2_dimension_heatmap.pdf")


def fig3_delta_heatmap():
    """Figure 3: System prompt delta heatmap (prompted - baseline)."""
    df = pd.read_csv("/Users/angelo.leone/Documents/pro-worker-benchmark/analysis_output/dimension_scores.csv")

    models_list = sorted(df["Model"].unique())
    dims = DIMENSIONS

    matrix = np.zeros((len(models_list), len(dims)))
    for i, model in enumerate(models_list):
        for j, dim in enumerate(dims):
            b = df[(df["Model"] == model) & (df["Variant"] == "baseline") & (df["Dimension"] == dim)]
            p = df[(df["Model"] == model) & (df["Variant"] == "with_system_prompt") & (df["Dimension"] == dim)]
            if not b.empty and not p.empty:
                matrix[i, j] = p.iloc[0]["Mean"] - b.iloc[0]["Mean"]

    # Sort by mean delta
    mean_deltas = matrix.mean(axis=1)
    order = np.argsort(-mean_deltas)
    matrix = matrix[order]
    models_sorted = [models_list[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=3)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in dims], rotation=45, ha="right")
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in models_sorted])

    for i in range(len(models_sorted)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val > 2.0 or val < -0.5 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, label="Δ Score (prompted − baseline)", shrink=0.8)
    ax.set_title("System Prompt Effect by Dimension and Model")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_delta_heatmap.pdf")
    plt.savefig(f"{OUTPUT_DIR}/fig3_delta_heatmap.png")
    plt.close()
    print("  fig3_delta_heatmap.pdf")


def fig4_radar_chart():
    """Figure 4: Radar chart comparing top 3 models (prompted)."""
    df = pd.read_csv("/Users/angelo.leone/Documents/pro-worker-benchmark/analysis_output/dimension_scores.csv")
    prompted = df[df["Variant"] == "with_system_prompt"]

    top_models = ["GLM 5.1", "Gemma 4 31B", "DeepSeek V3.2"]
    dims = [d for d in DIMENSIONS if d != "adversarial_resilience"]  # Skip AR (L3 only)
    n_dims = len(dims)

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = ["#1565C0", "#2E7D32", "#E65100"]
    for model, color in zip(top_models, colors):
        values = []
        for dim in dims:
            row = prompted[(prompted["Model"] == model) & (prompted["Dimension"] == dim)]
            values.append(row.iloc[0]["Mean"] if not row.empty else 0)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=MODEL_SHORT.get(model, model), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in dims], size=8)
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1", "2", "3"], size=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Top 3 Models: Dimension Profile (With System Prompt)", pad=20)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_radar_chart.pdf")
    plt.savefig(f"{OUTPUT_DIR}/fig4_radar_chart.png")
    plt.close()
    print("  fig4_radar_chart.pdf")


def fig5_effect_sizes():
    """Figure 5: Cohen's d effect sizes per dimension (averaged across models)."""
    df = pd.read_csv("/Users/angelo.leone/Documents/pro-worker-benchmark/analysis_output/effect_sizes.csv")

    dims = [d for d in DIMENSIONS if d in df["Dimension"].values]

    means = []
    stds = []
    for dim in dims:
        d_vals = df[df["Dimension"] == dim]["Cohens_d"].values
        means.append(np.mean(d_vals))
        stds.append(np.std(d_vals))

    # Sort by effect size
    order = np.argsort(means)[::-1]
    dims_sorted = [dims[i] for i in order]
    means_sorted = [means[i] for i in order]
    stds_sorted = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1565C0" if m > 0.8 else "#42A5F5" if m > 0.5 else "#90CAF9" for m in means_sorted]
    bars = ax.barh(range(len(dims_sorted)), means_sorted, xerr=stds_sorted,
                   color=colors, capsize=3, error_kw={"linewidth": 0.8})

    ax.set_yticks(range(len(dims_sorted)))
    ax.set_yticklabels([DIM_SHORT.get(d, d) for d in dims_sorted])
    ax.set_xlabel("Cohen's d (System Prompt Effect Size)")
    ax.axvline(x=0.2, color="gray", linestyle=":", linewidth=0.5, label="Small (0.2)")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.5, label="Medium (0.5)")
    ax.axvline(x=0.8, color="gray", linestyle="-", linewidth=0.5, label="Large (0.8)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("System Prompt Effect Size by Dimension")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_effect_sizes.pdf")
    plt.savefig(f"{OUTPUT_DIR}/fig5_effect_sizes.png")
    plt.close()
    print("  fig5_effect_sizes.pdf")


if __name__ == "__main__":
    print("Generating figures for Pro-Worker AI Benchmark v2.0 paper...")
    fig1_pwi_comparison()
    fig2_dimension_heatmap()
    fig3_delta_heatmap()
    fig4_radar_chart()
    fig5_effect_sizes()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
