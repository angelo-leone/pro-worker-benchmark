"""
Generate publication-quality figures for the Pro-Worker AI Benchmark paper.
Outputs PDF figures suitable for LaTeX inclusion.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------- paths ----------
RESULTS_DIR = Path(__file__).parent.parent / "pro-worker-benchmark" / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette - professional, colorblind-friendly
MODEL_COLORS = {
    "Llama 3.1 70B": "#2171B5",       # blue
    "Mistral Small 3.1 24B": "#CB181D", # red
    "Qwen 2.5 72B": "#238B45",         # green
}
BASELINE_ALPHA = 0.45
PROMPTED_ALPHA = 1.0

# ---------- data (from benchmark results) ----------
models = ["Llama 3.1 70B", "Mistral Small 3.1 24B", "Qwen 2.5 72B"]
short_models = ["Llama 70B", "Mistral 24B", "Qwen 72B"]

pwi_baseline = [40.9, 36.7, 37.4]
pwi_prompted = [61.0, 79.4, 62.7]
pwi_delta    = [20.1, 42.7, 25.3]

dims = ["CF", "CE", "SP", "DA", "UT", "CO", "AR"]
dim_full = [
    "Cognitive\nForcing", "Contrastive\nExplanation", "Skill\nPreservation",
    "Draft\nAnnotation", "Uncertainty\nTransparency", "Complementarity",
    "Adversarial\nResilience",
]

# baseline scores [model][dim]
baseline = {
    "Llama 3.1 70B":          [0.27, 1.93, 1.67, 1.07, 1.80, 1.07, 0.97],
    "Mistral Small 3.1 24B":  [0.13, 1.33, 1.73, 1.00, 1.80, 0.93, 1.03],
    "Qwen 2.5 72B":           [0.27, 1.27, 1.73, 1.00, 1.80, 1.00, 1.00],
}
prompted = {
    "Llama 3.1 70B":          [1.33, 2.53, 2.07, 1.47, 1.73, 1.87, 1.87],
    "Mistral Small 3.1 24B":  [2.53, 2.53, 2.33, 1.53, 2.13, 2.87, 2.41],
    "Qwen 2.5 72B":           [0.80, 2.21, 2.47, 1.80, 1.93, 2.27, 2.10],
}

# Layer 3 adversarial by category (with prompt only)
adv_categories = ["Direct", "Urgency", "Authority", "Emotional"]
adv_scores = {
    "Llama 3.1 70B":          [1.33, 2.25, 1.67, 3.00],
    "Mistral Small 3.1 24B":  [2.33, 2.50, 2.67, 3.00],
    "Qwen 2.5 72B":           [2.33, 2.00, 2.00, 3.00],
}


# ==========================================================================
# FIGURE 1 — PWI Comparison (Baseline vs Prompted)
# ==========================================================================
def fig1_pwi_comparison():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(models))
    w = 0.32

    bars_base = ax.bar(x - w/2, pwi_baseline, w, label="Baseline",
                       color=[MODEL_COLORS[m] for m in models],
                       alpha=BASELINE_ALPHA, edgecolor="white", linewidth=0.5)
    bars_prompt = ax.bar(x + w/2, pwi_prompted, w, label="With Prompt",
                         color=[MODEL_COLORS[m] for m in models],
                         alpha=PROMPTED_ALPHA, edgecolor="white", linewidth=0.5)

    # Delta annotations
    for i, (b, p, d) in enumerate(zip(pwi_baseline, pwi_prompted, pwi_delta)):
        ax.annotate(f"+{d:.1f}",
                    xy=(i + w/2, p), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#333333")

    # Midpoint line
    ax.axhline(y=50, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(len(models) - 0.6, 51, "Midpoint (50)", fontsize=7, color="#999999",
            ha="right")

    ax.set_ylabel("Pro-Worker Index (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_models)
    ax.set_ylim(0, 95)
    ax.set_xlim(-0.5, len(models) - 0.5)

    # Custom legend
    base_patch = mpatches.Patch(color="#888888", alpha=BASELINE_ALPHA, label="Baseline")
    prompt_patch = mpatches.Patch(color="#888888", alpha=PROMPTED_ALPHA, label="With Prompt")
    ax.legend(handles=[base_patch, prompt_patch], loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig1_pwi_comparison.pdf"
    fig.savefig(out)
    fig.savefig(FIGURES_DIR / "fig1_pwi_comparison.png")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# FIGURE 2 — Radar Chart: Per-Dimension Profile (Prompted)
# ==========================================================================
def fig2_radar_chart():
    categories = dims
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

    for model in models:
        values = prompted[model] + [prompted[model][0]]
        ax.plot(angles, values, linewidth=1.8, label=model.replace("Small 3.1", "3.1"),
                color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.08, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=8)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], fontsize=7, color="#666666")
    ax.set_rlabel_position(30)

    # Radial grid styling
    ax.grid(color="#cccccc", linewidth=0.5)
    ax.spines["polar"].set_color("#cccccc")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), frameon=True,
              framealpha=0.9, edgecolor="#cccccc", fontsize=7.5)

    out = FIGURES_DIR / "fig2_radar_prompted.pdf"
    fig.savefig(out)
    fig.savefig(FIGURES_DIR / "fig2_radar_prompted.png")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# FIGURE 3 — Heatmap: All Dimensions × Model/Variant
# ==========================================================================
def fig3_heatmap():
    row_labels = []
    data = []
    for model in models:
        short = model.split(" ")[0] if "Mistral" not in model else "Mistral"
        row_labels.append(f"{short} baseline")
        data.append(baseline[model])
        row_labels.append(f"{short} w/ prompt")
        data.append(prompted[model])

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(data_arr, annot=True, fmt=".2f", cmap=cmap,
                vmin=0, vmax=3, linewidths=0.6, linecolor="white",
                xticklabels=dims, yticklabels=row_labels,
                cbar_kws={"label": "Score (0–3)", "shrink": 0.8},
                ax=ax, annot_kws={"fontsize": 8})

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    # Add horizontal separators between models
    for i in [2, 4]:
        ax.axhline(y=i, color="black", linewidth=1.5)

    out = FIGURES_DIR / "fig3_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(FIGURES_DIR / "fig3_heatmap.png")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# FIGURE 4 — Adversarial Resilience by Pressure Category
# ==========================================================================
def fig4_adversarial():
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    x = np.arange(len(adv_categories))
    w = 0.22
    offsets = [-w, 0, w]

    for i, model in enumerate(models):
        short = model.split(" ")[0] if "Mistral" not in model else "Mistral"
        ax.bar(x + offsets[i], adv_scores[model], w,
               label=f"{short}",
               color=MODEL_COLORS[model], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Resilience Score (0–3)")
    ax.set_xticks(x)
    ax.set_xticklabels(adv_categories)
    ax.set_ylim(0, 3.4)
    ax.axhline(y=3.0, color="#cccccc", linewidth=0.5, linestyle=":")

    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc", ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig4_adversarial.pdf"
    fig.savefig(out)
    fig.savefig(FIGURES_DIR / "fig4_adversarial.png")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# FIGURE 5 — System Prompt Delta per Dimension
# ==========================================================================
def fig5_delta_chart():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    x = np.arange(len(dims))
    w = 0.22
    offsets = [-w, 0, w]

    for i, model in enumerate(models):
        deltas = [prompted[model][j] - baseline[model][j] for j in range(len(dims))]
        short = model.split(" ")[0] if "Mistral" not in model else "Mistral"
        bars = ax.bar(x + offsets[i], deltas, w,
                      label=f"{short}",
                      color=MODEL_COLORS[model], alpha=0.85,
                      edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score Improvement (Δ)")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.axhline(y=0, color="black", linewidth=0.6)

    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc", ncol=3,
              loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.3, 2.8)

    out = FIGURES_DIR / "fig5_delta_dimensions.pdf"
    fig.savefig(out)
    fig.savefig(FIGURES_DIR / "fig5_delta_dimensions.png")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    print("Generating figures for Pro-Worker AI Benchmark paper...\n")

    print("[1/5] PWI comparison chart")
    fig1_pwi_comparison()

    print("[2/5] Radar chart (prompted profiles)")
    fig2_radar_chart()

    print("[3/5] Heatmap (all dimensions × model/variant)")
    fig3_heatmap()

    print("[4/5] Adversarial resilience breakdown")
    fig4_adversarial()

    print("[5/5] System prompt delta per dimension")
    fig5_delta_chart()

    print(f"\nAll figures saved to {FIGURES_DIR}")
