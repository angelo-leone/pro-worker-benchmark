"""
Streamlit dashboard for visualizing Pro-Worker AI Benchmark results.

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.analysis import (
    load_all_results,
    analyze_single_run,
    compare_models,
    compute_system_prompt_delta,
    DEFAULT_WEIGHTS,
    MAX_SCORE,
)


RESULTS_DIR = Path(__file__).parent / "results"

# Dimension display names
DIM_LABELS = {
    "cognitive_forcing": "Cognitive Forcing",
    "contrastive_explanation": "Contrastive Explanation",
    "skill_preservation": "Skill Preservation",
    "draft_annotation": "Draft Annotation",
    "uncertainty_transparency": "Uncertainty Transparency",
    "complementarity": "Complementarity",
    "adversarial_resilience": "Adversarial Resilience",
}


def load_data():
    """Load all results, handling empty directory."""
    if not RESULTS_DIR.exists():
        return []
    return load_all_results(RESULTS_DIR)


def render_pwi_comparison(comparison_df: pd.DataFrame):
    """Render the main PWI comparison bar chart."""
    if comparison_df.empty:
        st.warning("No data to display.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(comparison_df) * 0.6)))

    labels = [
        f"{row['model']}\n({row['variant']})"
        for _, row in comparison_df.iterrows()
    ]
    values = comparison_df["pwi"].values

    colors = []
    for v in values:
        if v >= 70:
            colors.append("#2ecc71")  # green
        elif v >= 50:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Pro-Worker Index (PWI)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title("Pro-Worker Index by Model", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=10, fontweight="bold",
        )

    ax.axvline(x=70, color="#2ecc71", linestyle="--", alpha=0.5, label="Strong (70+)")
    ax.axvline(x=50, color="#f39c12", linestyle="--", alpha=0.5, label="Moderate (50+)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_radar_chart(analysis: dict, title: str):
    """Render a radar/spider chart for per-dimension scores."""
    if "layer1" not in analysis:
        return

    dimensions = list(analysis["layer1"].keys())
    if not dimensions:
        return

    values = [analysis["layer1"][d]["mean"] / MAX_SCORE for d in dimensions]
    labels = [DIM_LABELS.get(d, d) for d in dimensions]

    # Add adversarial if available
    if "layer3" in analysis:
        dimensions.append("adversarial_resilience")
        values.append(analysis["layer3"]["mean"] / MAX_SCORE)
        labels.append("Adversarial Resilience")

    # Close the polygon
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="#3498db")
    ax.plot(angles, values, "o-", color="#3498db", linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.75", "1.50", "2.25", "3.00"], fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_system_prompt_delta(delta_df: pd.DataFrame):
    """Render a chart showing the impact of the system prompt."""
    if delta_df.empty:
        st.info("No baseline vs. system prompt comparison available.")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(delta_df) * 0.8)))

    for i, (_, row) in enumerate(delta_df.iterrows()):
        # Baseline bar
        ax.barh(
            i, row["pwi_baseline"], color="#bdc3c7", edgecolor="white",
            label="Baseline" if i == 0 else "",
        )
        # With prompt bar (overlay)
        ax.barh(
            i, row["pwi_with_prompt"], color="#3498db", alpha=0.7,
            edgecolor="white",
            label="With System Prompt" if i == 0 else "",
        )
        # Delta label
        delta = row["pwi_delta"]
        sign = "+" if delta >= 0 else ""
        color = "#2ecc71" if delta >= 0 else "#e74c3c"
        ax.text(
            max(row["pwi_baseline"], row["pwi_with_prompt"]) + 2,
            i, f"{sign}{delta:.1f}",
            va="center", fontsize=11, fontweight="bold", color=color,
        )

    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(delta_df["model"].values, fontsize=10)
    ax.set_xlabel("Pro-Worker Index (PWI)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_title("System Prompt Impact on PWI", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_dimension_heatmap(comparison_df: pd.DataFrame):
    """Render a heatmap of per-dimension scores across models."""
    dim_cols = [c for c in comparison_df.columns if c.startswith("l1_")]
    if not dim_cols:
        return

    data = comparison_df[dim_cols].copy()
    data.columns = [DIM_LABELS.get(c[3:], c[3:]) for c in dim_cols]
    data.index = [
        f"{row['model']} ({row['variant']})"
        for _, row in comparison_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(12, max(3, len(data) * 0.6)))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)

    # Add text annotations
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            color = "white" if val < 1.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Per-Dimension Scores (0-3)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_adversarial_breakdown(results: list[dict]):
    """Render adversarial resilience breakdown by category."""
    all_categories = {}
    for r in results:
        if "layer3" not in r:
            continue
        label = f"{r['model_name']} ({r['variant']})"
        for item in r["layer3"]:
            cat = item.get("category", "unknown")
            if item["score"] >= 0:
                if cat not in all_categories:
                    all_categories[cat] = {}
                if label not in all_categories[cat]:
                    all_categories[cat][label] = []
                all_categories[cat][label].append(item["score"])

    if not all_categories:
        return

    # Build DataFrame
    models = sorted({
        f"{r['model_name']} ({r['variant']})" for r in results
    })
    rows = []
    for cat in sorted(all_categories.keys()):
        row = {"category": cat}
        for model in models:
            scores = all_categories[cat].get(model, [])
            row[model] = sum(scores) / len(scores) if scores else 0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("category")

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.5)))
    df.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Average Score (0-3)", fontsize=11)
    ax.set_title("Adversarial Resilience by Pressure Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 3.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_scenario_details(results: dict):
    """Render detailed scenario results for a single model run."""
    if "layer2" not in results:
        st.info("No Layer 2 scenario data.")
        return

    for scenario in results["layer2"]:
        with st.expander(
            f"Scenario: {scenario['scenario_id']} "
            f"(avg={scenario['average_score']:.2f}, "
            f"behaviors={scenario['behavior_pass_rate']:.0%})"
        ):
            for turn in scenario["turns"]:
                st.markdown(f"**Turn {turn['turn']}**")
                st.markdown(f"*User:* {turn['user_message'][:200]}...")
                st.markdown(f"*AI Response:* {turn['ai_response'][:300]}...")

                # Dimension scores
                for dim, scores in turn["dimension_scores"].items():
                    st.markdown(
                        f"  - {DIM_LABELS.get(dim, dim)}: "
                        f"**{scores['score']}/3** — {scores['reasoning']}"
                    )

                # Behavior results
                if turn["behavior_results"]:
                    behaviors = turn["behavior_results"]
                    passed = sum(1 for v in behaviors.values() if v is True)
                    total = sum(1 for v in behaviors.values() if v is not None)
                    st.markdown(f"  - Behaviors: **{passed}/{total}** passed")

                st.markdown("---")


# ========== MAIN APP ==========

def main():
    st.set_page_config(
        page_title="Pro-Worker AI Benchmark",
        page_icon="🏗️",
        layout="wide",
    )

    st.title("Pro-Worker AI Benchmark Dashboard")
    st.markdown(
        "Evaluating LLMs on whether they **augment** human workers "
        "rather than replacing them."
    )

    # Load data
    all_results = load_data()

    if not all_results:
        st.warning(
            "No results found. Run the benchmark first:\n\n"
            "```\npython -m src.runner\n```"
        )
        st.stop()

    # Sidebar: model selector
    st.sidebar.header("Filters")
    model_labels = [
        f"{r.get('model_name', 'unknown')} ({r.get('variant', 'unknown')})"
        for r in all_results
    ]
    selected = st.sidebar.multiselect(
        "Select models to compare",
        options=model_labels,
        default=model_labels,
    )

    filtered = [
        r for r, label in zip(all_results, model_labels)
        if label in selected
    ]

    if not filtered:
        st.warning("Select at least one model.")
        st.stop()

    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Dimensions", "Scenarios", "Adversarial", "System Prompt Impact",
    ])

    comparison = compare_models(filtered)

    # Tab 1: Overview
    with tab1:
        st.header("Pro-Worker Index (PWI) Comparison")
        render_pwi_comparison(comparison)

        # Summary metrics
        st.subheader("Summary")
        cols = st.columns(len(filtered))
        for i, (r, label) in enumerate(zip(filtered, selected)):
            analysis = analyze_single_run(r)
            with cols[i % len(cols)]:
                st.metric(label=label, value=f"{analysis['pwi']:.1f}/100")

    # Tab 2: Per-Dimension breakdown
    with tab2:
        st.header("Per-Dimension Scores")
        render_dimension_heatmap(comparison)

        st.subheader("Radar Charts")
        cols = st.columns(min(3, len(filtered)))
        for i, r in enumerate(filtered):
            analysis = analyze_single_run(r)
            label = f"{r.get('model_name', '?')} ({r.get('variant', '?')})"
            with cols[i % len(cols)]:
                render_radar_chart(analysis, label)

    # Tab 3: Scenario Details
    with tab3:
        st.header("Multi-Turn Scenario Results")
        selected_model = st.selectbox(
            "Select model run", options=selected
        )
        idx = selected.index(selected_model) if selected_model in selected else 0
        render_scenario_details(filtered[idx])

    # Tab 4: Adversarial Analysis
    with tab4:
        st.header("Adversarial Resilience Analysis")
        render_adversarial_breakdown(filtered)

    # Tab 5: System Prompt Impact
    with tab5:
        st.header("Impact of Pro-Worker System Prompt")
        delta = compute_system_prompt_delta(filtered)
        render_system_prompt_delta(delta)

        if not delta.empty:
            st.subheader("Per-Dimension Delta")
            delta_cols = [c for c in delta.columns if c.endswith("_delta") and c != "pwi_delta"]
            if delta_cols:
                delta_display = delta[["model"] + delta_cols].copy()
                delta_display.columns = ["Model"] + [
                    DIM_LABELS.get(c.replace("_delta", ""), c.replace("_delta", ""))
                    for c in delta_cols
                ]
                st.dataframe(delta_display, use_container_width=True)


if __name__ == "__main__":
    main()
