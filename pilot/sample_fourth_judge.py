"""
Stratified sampler for the Claude-as-fourth-judge analysis.

Draws N=500 Layer-1 instances stratified by dimension and balanced
across model x condition cells. Writes a JSONL with everything needed
to score each item: prompt, response, dimension, panel scores.

Run from repo root:
  python pilot/sample_fourth_judge.py
"""

from __future__ import annotations

import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "dataset_staging" / "results"
OUT_PATH = ROOT / "pilot" / "fourth_judge_items.jsonl"

TARGET_N = 500
PER_DIMENSION = TARGET_N // 10  # 50 per dimension across 10 L1 dims
SEED = 4242  # Distinct from the human-pilot seed (42) to keep the two samples independent.

DIMENSIONS = [
    "uncertainty_transparency",
    "appropriate_reliance",
    "metacognitive_calibration",
    "cognitive_forcing",
    "anti_sycophancy",
    "complementarity",
    "ethical_surfacing",
    "contrastive_explanation",
    "skill_preservation",
    "draft_annotation",
]


def latest_results_per_combo() -> dict[tuple[str, str], Path]:
    by_key: dict[tuple[str, str], Path] = {}
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.loads(Path(f).read_text())
        if not all(k in d for k in ("layer1", "layer2", "layer3")):
            continue
        by_key[(d["model_name"], d["variant"])] = Path(f)
    return by_key


def build_pool() -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = defaultdict(list)
    for (model, variant), path in latest_results_per_combo().items():
        d = json.loads(path.read_text())
        for dim, prompts in d["layer1"].items():
            for p in prompts:
                for run_idx, run in enumerate(p.get("runs", [])):
                    if "response" not in run:
                        continue
                    scores = run.get("individual_scores")
                    if not scores:
                        continue
                    panel_median = int(statistics.median(scores))
                    pool[dim].append(
                        {
                            "dimension": dim,
                            "model": model,
                            "condition": variant,
                            "prompt_id": p["prompt_id"],
                            "run_idx": run_idx,
                            "user_prompt": p["prompt"],
                            "response": run["response"],
                            "panel_individual_scores": list(scores),
                            "panel_median": panel_median,
                        }
                    )
    return pool


def stratified_sample(pool: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for dim in DIMENSIONS:
        candidates = [c for c in pool[dim] if c["run_idx"] == 0]
        if len(candidates) < PER_DIMENSION:
            extras = [c for c in pool[dim] if c["run_idx"] != 0]
            rng.shuffle(extras)
            candidates = candidates + extras

        base = [c for c in candidates if c["condition"] == "baseline"]
        prompt = [c for c in candidates if c["condition"] == "with_system_prompt"]

        target_each = PER_DIMENSION // 2
        rng.shuffle(base)
        rng.shuffle(prompt)
        out.extend(base[:target_each])
        out.extend(prompt[:target_each])
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)
    pool = build_pool()
    sample = stratified_sample(pool, rng)
    if len(sample) != TARGET_N:
        raise SystemExit(f"Got {len(sample)} items, expected {TARGET_N}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w") as f:
        for i, item in enumerate(sample, start=1):
            item_out = {"item_id": i, **item}
            f.write(json.dumps(item_out) + "\n")

    print(f"Wrote {len(sample)} items to {OUT_PATH}")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s in sample:
        counts[(s["dimension"], s["condition"])] += 1
    print("\nItems per (dimension, condition):")
    for dim in DIMENSIONS:
        b = counts[(dim, "baseline")]
        p = counts[(dim, "with_system_prompt")]
        print(f"  {dim:30s} baseline={b}  prompted={p}  total={b+p}")


if __name__ == "__main__":
    main()
