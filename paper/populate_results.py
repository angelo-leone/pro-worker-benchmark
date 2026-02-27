"""
Populate the LaTeX paper with actual benchmark results.
Run this after the benchmark completes to fill in all '---' placeholders
in the LaTeX tables with real data.

Usage:
    python populate_results.py
"""

import json
import sys
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "pro-worker-benchmark" / "results"
PAPER_PATH = Path(__file__).parent / "pro_worker_benchmark.tex"

sys.path.insert(0, str(Path(__file__).parent.parent / "pro-worker-benchmark"))
from src.analysis import (
    load_all_results,
    analyze_single_run,
    compute_system_prompt_delta,
)


def load_and_analyze():
    """Load all results and compute analyses."""
    results = load_all_results(RESULTS_DIR)
    if not results:
        print("No results found in", RESULTS_DIR)
        return None

    analyses = {}
    for r in results:
        key = f"{r.get('model_name', 'unknown')}_{r.get('variant', 'unknown')}"
        analyses[key] = analyze_single_run(r)

    return results, analyses


def format_score(val, decimals=2):
    """Format a score value, returning '---' if missing."""
    if val is None or val == 0:
        return "---"
    return f"{val:.{decimals}f}"


def build_pwi_table(analyses):
    """Build the PWI comparison table rows."""
    models = [
        ("Llama 3.1 70B", "Llama 3.1 70B"),
        ("Mistral Small 3.1 24B", "Mistral Small 3.1 24B"),
        ("Qwen 2.5 72B", "Qwen 2.5 72B"),
    ]

    rows = []
    for display_name, model_name in models:
        base_key = f"{model_name}_baseline"
        prompt_key = f"{model_name}_with_system_prompt"

        base_pwi = analyses.get(base_key, {}).get("pwi")
        prompt_pwi = analyses.get(prompt_key, {}).get("pwi")

        if base_pwi is not None and prompt_pwi is not None:
            delta = prompt_pwi - base_pwi
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            row = f"{display_name} & {base_pwi:.1f} & {prompt_pwi:.1f} & {delta_str}"
        elif base_pwi is not None:
            row = f"{display_name} & {base_pwi:.1f} & --- & ---"
        elif prompt_pwi is not None:
            row = f"{display_name} & --- & {prompt_pwi:.1f} & ---"
        else:
            row = f"{display_name} & --- & --- & ---"

        rows.append(row)
    return rows


def build_dimension_table(analyses):
    """Build per-dimension score table rows."""
    models = [
        ("Llama 3.1 70B", "Llama 3.1 70B"),
        ("Mistral Small 3.1 24B", "Mistral Small 3.1 24B"),
        ("Qwen 2.5 72B", "Qwen 2.5 72B"),
    ]
    dims = [
        "cognitive_forcing", "contrastive_explanation", "skill_preservation",
        "draft_annotation", "uncertainty_transparency", "complementarity",
    ]

    rows = []
    for display_name, model_name in models:
        for variant_label, variant_key in [("baseline", "baseline"), ("w/ prompt", "with_system_prompt")]:
            key = f"{model_name}_{variant_key}"
            analysis = analyses.get(key, {})

            # Layer 1 dimensions
            l1 = analysis.get("layer1", {})
            dim_scores = []
            for d in dims:
                score = l1.get(d, {}).get("mean")
                dim_scores.append(format_score(score))

            # Layer 3 adversarial
            l3 = analysis.get("layer3", {})
            ar_score = l3.get("mean") if l3 else None
            dim_scores.append(format_score(ar_score))

            label = display_name if variant_label == "baseline" else ""
            row = f"{label} & {variant_label} & " + " & ".join(dim_scores)
            rows.append(row)

    return rows


def build_layer2_table(analyses):
    """Build Layer 2 multi-turn scenario table rows."""
    models = [
        ("Llama 3.1 70B", "Llama 3.1 70B"),
        ("Mistral Small 3.1 24B", "Mistral Small 3.1 24B"),
        ("Qwen 2.5 72B", "Qwen 2.5 72B"),
    ]

    rows = []
    for display_name, model_name in models:
        base_key = f"{model_name}_baseline"
        prompt_key = f"{model_name}_with_system_prompt"

        base_l2 = analyses.get(base_key, {}).get("layer2", {})
        prompt_l2 = analyses.get(prompt_key, {}).get("layer2", {})

        base_avg = format_score(base_l2.get("overall_average_score"))
        prompt_avg = format_score(prompt_l2.get("overall_average_score"))
        base_beh = format_score(base_l2.get("overall_behavior_pass_rate"), 0) if base_l2.get("overall_behavior_pass_rate") else "---"
        prompt_beh = format_score(prompt_l2.get("overall_behavior_pass_rate"), 0) if prompt_l2.get("overall_behavior_pass_rate") else "---"

        # Format behavior rates as percentages
        if base_beh != "---":
            base_beh = f"{base_l2.get('overall_behavior_pass_rate', 0) * 100:.0f}\\%"
        if prompt_beh != "---":
            prompt_beh = f"{prompt_l2.get('overall_behavior_pass_rate', 0) * 100:.0f}\\%"

        row = f"{display_name} & {base_avg} & {prompt_avg} & {base_beh} & {prompt_beh}"
        rows.append(row)
    return rows


def build_layer3_table(analyses):
    """Build Layer 3 adversarial breakdown table rows."""
    models = [
        ("Llama 3.1 70B", "Llama 3.1 70B"),
        ("Mistral Small 3.1 24B", "Mistral Small 3.1 24B"),
        ("Qwen 2.5 72B", "Qwen 2.5 72B"),
    ]
    categories = ["direct_pressure", "urgency", "authority", "emotional"]

    rows = []
    for display_name, model_name in models:
        prompt_key = f"{model_name}_with_system_prompt"
        analysis = analyses.get(prompt_key, {})
        l3 = analysis.get("layer3", {})
        by_cat = l3.get("by_category", {})

        cat_scores = []
        for cat in categories:
            score = by_cat.get(cat, {}).get("mean")
            cat_scores.append(format_score(score))

        row = f"{display_name} & " + " & ".join(cat_scores)
        rows.append(row)
    return rows


def replace_table_rows(tex_content, table_label, new_rows, separator=" \\\\\n"):
    """Replace table rows in LaTeX content between midrule and bottomrule."""
    # This is a simplified approach - find the table and replace the data rows
    # For now, we'll print what should go in
    return new_rows


def main():
    print("Loading benchmark results...")
    result = load_and_analyze()
    if result is None:
        return

    results, analyses = result

    print(f"\nFound {len(results)} result files:")
    for r in results:
        print(f"  - {r.get('model_name')} ({r.get('variant')})")

    print("\n" + "=" * 60)
    print("PWI SCORES (Table 2)")
    print("=" * 60)
    for row in build_pwi_table(analyses):
        print(f"  {row} \\\\")

    print("\n" + "=" * 60)
    print("DIMENSION SCORES (Table 3)")
    print("=" * 60)
    dim_rows = build_dimension_table(analyses)
    for i, row in enumerate(dim_rows):
        suffix = " \\\\" if (i + 1) % 2 != 0 else " \\\\\n\\midrule"
        print(f"  {row}{suffix}")

    print("\n" + "=" * 60)
    print("LAYER 2 SCENARIOS (Table 4)")
    print("=" * 60)
    for row in build_layer2_table(analyses):
        print(f"  {row} \\\\")

    print("\n" + "=" * 60)
    print("LAYER 3 ADVERSARIAL (Table 5)")
    print("=" * 60)
    for row in build_layer3_table(analyses):
        print(f"  {row} \\\\")

    # Print full analysis summary
    print("\n" + "=" * 60)
    print("FULL ANALYSIS SUMMARY")
    print("=" * 60)
    for key, analysis in sorted(analyses.items()):
        print(f"\n{key}:")
        print(f"  PWI: {analysis.get('pwi', 0):.1f}")
        if "layer1" in analysis:
            for dim, stats in analysis["layer1"].items():
                print(f"  L1 {dim}: mean={stats['mean']:.2f}, n={stats['n']}")
        if "layer2" in analysis:
            l2 = analysis["layer2"]
            print(f"  L2 avg_score: {l2.get('overall_average_score', 0):.2f}")
            print(f"  L2 behavior_rate: {l2.get('overall_behavior_pass_rate', 0):.2%}")
        if "layer3" in analysis:
            l3 = analysis["layer3"]
            print(f"  L3 adversarial: mean={l3.get('mean', 0):.2f}, n={l3.get('n', 0)}")


if __name__ == "__main__":
    main()
