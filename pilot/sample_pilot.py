"""
Stratified sampler for the human validation pilot.

Reads the 14 (model, variant) result files in dataset_staging/results/, builds a
pool of Layer-1 items, draws a stratified N=30 sample with seed=42, and writes:

  pilot/pilot_items.md          author-facing scoring sheet (no LLM scores)
  pilot/pilot_ground_truth.json analysis-side reference (LLM-judge medians)
  pilot/pilot_scores.csv        empty scores CSV for the author to fill in

Run from repo root:
  python pilot/sample_pilot.py
"""

from __future__ import annotations

import csv
import glob
import json
import random
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "dataset_staging" / "results"
RUBRICS_PATH = ROOT / "rubrics" / "dimension_rubrics.yaml"
OUT_DIR = ROOT / "pilot"

ALLOCATION = {
    "uncertainty_transparency": 6,
    "appropriate_reliance": 6,
    "metacognitive_calibration": 4,
    "cognitive_forcing": 3,
    "anti_sycophancy": 3,
    "complementarity": 2,
    "ethical_surfacing": 2,
    "contrastive_explanation": 2,
    "skill_preservation": 1,
    "draft_annotation": 1,
}

DIMENSION_DISPLAY = {
    "uncertainty_transparency": "Uncertainty Transparency",
    "appropriate_reliance": "Appropriate Reliance",
    "metacognitive_calibration": "Metacognitive Calibration",
    "cognitive_forcing": "Cognitive Forcing",
    "anti_sycophancy": "Anti-Sycophancy",
    "complementarity": "Complementarity",
    "ethical_surfacing": "Ethical Surfacing",
    "contrastive_explanation": "Contrastive Explanation",
    "skill_preservation": "Skill Preservation",
    "draft_annotation": "Draft with Annotations",
}


def latest_results_per_combo() -> dict[tuple[str, str], Path]:
    """Pick the latest result file for each (model, variant) that has all 3 layers."""
    by_key: dict[tuple[str, str], Path] = {}
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.loads(Path(f).read_text())
        if not all(k in d for k in ("layer1", "layer2", "layer3")):
            continue
        by_key[(d["model_name"], d["variant"])] = Path(f)
    return by_key


def build_pool() -> dict[str, list[dict]]:
    """For each Layer-1 dimension, collect candidate items across models and variants."""
    pool: dict[str, list[dict]] = defaultdict(list)
    for (model, variant), path in latest_results_per_combo().items():
        d = json.loads(path.read_text())
        for dim, prompts in d["layer1"].items():
            for p in prompts:
                for run_idx, run in enumerate(p.get("runs", [])):
                    if "response" not in run:
                        continue
                    scores = run.get("individual_scores")
                    if not scores or len(scores) < 1:
                        continue
                    judge_median = int(statistics.median(scores))
                    pool[dim].append(
                        {
                            "dimension": dim,
                            "model": model,
                            "condition": variant,
                            "prompt_id": p["prompt_id"],
                            "run_idx": run_idx,
                            "user_prompt": p["prompt"],
                            "response": run["response"],
                            "llm_judge_median": judge_median,
                            "llm_individual_scores": list(scores),
                        }
                    )
    return pool


def stratified_sample(pool: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    """Draw the N=30 sample per ALLOCATION, balancing baseline and prompted runs."""
    out: list[dict] = []
    for dim, n in ALLOCATION.items():
        candidates = pool[dim]

        run0 = [c for c in candidates if c["run_idx"] == 0]
        run1 = [c for c in candidates if c["run_idx"] == 1]
        # Prefer run 0 globally; only fall back to run 1 if run 0 is empty for a slot.

        n_base = n // 2
        n_prompt = n - n_base

        def pick(condition: str, k: int) -> list[dict]:
            primary = [c for c in run0 if c["condition"] == condition]
            if len(primary) >= k:
                # Prefer one prompt per model — diversify model coverage within the slot.
                rng.shuffle(primary)
                seen_models: set[str] = set()
                ordered: list[dict] = []
                tail: list[dict] = []
                for c in primary:
                    if c["model"] not in seen_models:
                        ordered.append(c)
                        seen_models.add(c["model"])
                    else:
                        tail.append(c)
                pool_for_pick = ordered + tail
                return pool_for_pick[:k]
            # Fall back: pad with run 1 candidates of the same condition.
            extras = [c for c in run1 if c["condition"] == condition]
            combined = primary + extras
            rng.shuffle(combined)
            return combined[:k]

        out.extend(pick("baseline", n_base))
        out.extend(pick("with_system_prompt", n_prompt))
    return out


def load_rubrics() -> dict[str, dict[int, dict[str, str]]]:
    raw = yaml.safe_load(RUBRICS_PATH.read_text())
    rubrics = {}
    for dim, body in raw.items():
        rubric = body.get("rubric", {})
        rubrics[dim] = {
            int(level): {
                "label": entry.get("label", ""),
                "description": " ".join((entry.get("description", "") or "").split()),
            }
            for level, entry in rubric.items()
        }
    return rubrics


def render_items_md(sample: list[dict], rubrics: dict[str, dict[int, dict[str, str]]]) -> str:
    out: list[str] = []
    out.append("# Pilot Items — Single-Annotator Validation\n")
    out.append(
        "Score each item on a 0-3 scale using the rubric anchors shown. "
        "Write your score and an optional one-line note in `pilot_scores.csv`. "
        "Do NOT open `pilot_ground_truth.json` until after you finish scoring all 30.\n"
    )
    for i, item in enumerate(sample, start=1):
        dim = item["dimension"]
        out.append(f"\n---\n\n## Item {i} (Dimension: {DIMENSION_DISPLAY[dim]})\n")
        out.append(f"**Prompt ID:** `{item['prompt_id']}`  \n")
        out.append("**User prompt:**\n\n")
        for line in item["user_prompt"].splitlines() or [""]:
            out.append(f"> {line}\n")
        out.append("\n**Model response:**\n\n")
        for line in item["response"].splitlines() or [""]:
            out.append(f"> {line}\n")
        out.append(f"\n**Rubric for {DIMENSION_DISPLAY[dim]}:**\n\n")
        for level in (3, 2, 1, 0):
            anchor = rubrics[dim].get(level, {})
            label = anchor.get("label", "")
            desc = anchor.get("description", "")
            wrapped = textwrap.fill(desc, width=110, subsequent_indent="  ")
            out.append(f"- **{level} ({label}):** {wrapped}\n")
        out.append("\n**Your score (0-3):** ___  \n")
        out.append("**Your note (optional, ~10 words):** ___\n")
    return "".join(out)


def render_ground_truth(sample: list[dict]) -> list[dict]:
    return [
        {
            "item_id": i,
            "dimension": item["dimension"],
            "model": item["model"],
            "condition": item["condition"],
            "prompt_id": item["prompt_id"],
            "run_idx": item["run_idx"],
            "llm_judge_median": item["llm_judge_median"],
            "llm_individual_scores": item["llm_individual_scores"],
        }
        for i, item in enumerate(sample, start=1)
    ]


def write_scores_csv(n: int, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "human_score", "note"])
        for i in range(1, n + 1):
            w.writerow([i, "", ""])


def main() -> None:
    rng = random.Random(42)
    pool = build_pool()

    missing = [dim for dim, n in ALLOCATION.items() if len(pool.get(dim, [])) < n]
    if missing:
        raise SystemExit(f"Pool missing dimensions or short on candidates: {missing}")

    sample = stratified_sample(pool, rng)
    if len(sample) != sum(ALLOCATION.values()):
        raise SystemExit(f"Sampled {len(sample)} items, expected {sum(ALLOCATION.values())}")

    rubrics = load_rubrics()
    OUT_DIR.mkdir(exist_ok=True)

    items_md = render_items_md(sample, rubrics)
    (OUT_DIR / "pilot_items.md").write_text(items_md)

    gt = render_ground_truth(sample)
    (OUT_DIR / "pilot_ground_truth.json").write_text(json.dumps(gt, indent=2))

    write_scores_csv(len(sample), OUT_DIR / "pilot_scores.csv")

    print(f"Wrote {len(sample)} items to:")
    print(f"  {OUT_DIR/'pilot_items.md'}")
    print(f"  {OUT_DIR/'pilot_ground_truth.json'}")
    print(f"  {OUT_DIR/'pilot_scores.csv'}")
    print()
    print("Distribution by dimension and condition:")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s in sample:
        counts[(s["dimension"], s["condition"])] += 1
    for dim in ALLOCATION:
        b = counts[(dim, "baseline")]
        p = counts[(dim, "with_system_prompt")]
        print(f"  {dim:30s} baseline={b}  prompted={p}  total={b+p}")


if __name__ == "__main__":
    main()
