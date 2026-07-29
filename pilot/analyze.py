"""
Agreement analysis for the human-validation pilot AND the Claude-as-fourth-judge run.

Both compute the same metrics — exact agreement, adjacent agreement,
quadratic-weighted Cohen's κ, mean absolute deviation — overall and per-dimension.

Usage:
  python pilot/analyze.py human      # human (N=30) vs panel median
  python pilot/analyze.py fourth     # Claude (N=500) vs panel median
  python pilot/analyze.py both       # both, when both inputs are available
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = ROOT / "pilot"


def quadratic_weighted_kappa(y_a: list[int], y_b: list[int], levels: int = 4) -> float:
    """Cohen's quadratic-weighted κ on a 0..levels-1 ordinal scale."""
    n = len(y_a)
    if n == 0:
        return float("nan")

    obs = [[0] * levels for _ in range(levels)]
    hist_a = [0] * levels
    hist_b = [0] * levels
    for a, b in zip(y_a, y_b):
        obs[a][b] += 1
        hist_a[a] += 1
        hist_b[b] += 1

    weights = [
        [((i - j) ** 2) / ((levels - 1) ** 2) for j in range(levels)] for i in range(levels)
    ]
    expected = [[(hist_a[i] * hist_b[j]) / n for j in range(levels)] for i in range(levels)]

    num = sum(weights[i][j] * obs[i][j] for i in range(levels) for j in range(levels))
    den = sum(weights[i][j] * expected[i][j] for i in range(levels) for j in range(levels))
    if den == 0:
        return float("nan")
    return 1 - num / den


def compute_metrics(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"n": 0, "exact": float("nan"), "adjacent": float("nan"),
                "mad": float("nan"), "qw_kappa": float("nan")}
    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b) / n
    adjacent = sum(1 for a, b in pairs if abs(a - b) <= 1) / n
    mad = sum(abs(a - b) for a, b in pairs) / n
    qwk = quadratic_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": n, "exact": exact, "adjacent": adjacent, "mad": mad, "qw_kappa": qwk}


def _per_dimension(by_dim: dict[str, list[tuple[int, int]]]) -> list[dict]:
    rows = []
    for dim, pairs in sorted(by_dim.items()):
        m = compute_metrics(pairs)
        rows.append({"dimension": dim, **m})
    return rows


def _format_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols = [(key, header, fmt), ...] where fmt is a Python format spec."""
    headers = [c[1] for c in cols]
    widths = [len(h) for h in headers]
    formatted_rows: list[list[str]] = []
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                s = "—"
            elif isinstance(v, float):
                s = format(v, fmt)
            else:
                s = str(v)
            cells.append(s)
        formatted_rows.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for cells in formatted_rows:
        out.append("| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |")
    return "\n".join(out)


def analyze_human() -> str | None:
    gt_path = PILOT_DIR / "pilot_ground_truth.json"
    sc_path = PILOT_DIR / "pilot_scores.csv"
    if not gt_path.exists() or not sc_path.exists():
        return None
    gt = {entry["item_id"]: entry for entry in json.loads(gt_path.read_text())}
    pairs: list[tuple[int, int]] = []
    by_dim: dict[str, list[tuple[int, int]]] = defaultdict(list)
    skipped = 0
    notes: dict[int, str] = {}
    import csv
    with sc_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                item_id = int(row["item_id"])
                hs_raw = row.get("human_score", "").strip()
                if hs_raw == "":
                    skipped += 1
                    continue
                hs = int(hs_raw)
            except (ValueError, KeyError):
                skipped += 1
                continue
            if hs < 0 or hs > 3:
                skipped += 1
                continue
            entry = gt.get(item_id)
            if entry is None:
                skipped += 1
                continue
            panel = entry["llm_judge_median"]
            pairs.append((hs, panel))
            by_dim[entry["dimension"]].append((hs, panel))
            notes[item_id] = (row.get("note") or "").strip()

    overall = compute_metrics(pairs)
    sub_keys = ("uncertainty_transparency", "appropriate_reliance")
    sub_pairs: list[tuple[int, int]] = []
    for k in sub_keys:
        sub_pairs.extend(by_dim.get(k, []))
    sub = compute_metrics(sub_pairs)

    rows = _per_dimension(by_dim)
    table = _format_table(
        rows,
        [
            ("dimension", "Dimension", ""),
            ("n", "N", ""),
            ("exact", "Exact %", ".1%"),
            ("adjacent", "Adjacent %", ".1%"),
            ("mad", "MAD", ".2f"),
            ("qw_kappa", "QW-κ", ".3f"),
        ],
    )

    out = []
    out.append("# Human-Pilot Agreement (single annotator vs LLM-judge panel median)\n")
    out.append(f"Items scored: **{overall['n']}**  (skipped {skipped})\n")
    out.append("\n## Headline numbers\n")
    out.append(f"- Exact agreement: **{overall['exact']:.1%}**")
    out.append(f"- Adjacent agreement (±1): **{overall['adjacent']:.1%}**")
    out.append(f"- Mean absolute deviation: **{overall['mad']:.2f}**")
    out.append(f"- Quadratic-weighted Cohen's κ: **{overall['qw_kappa']:.3f}**")
    out.append("\n## Lowest-IRR subset (UT + AR)\n")
    out.append(f"- N: {sub['n']}")
    out.append(f"- Exact: **{sub['exact']:.1%}**" if sub["n"] else "- (no items in subset)")
    out.append(f"- Adjacent (±1): **{sub['adjacent']:.1%}**" if sub["n"] else "")
    out.append(f"- MAD: **{sub['mad']:.2f}**" if sub["n"] else "")
    out.append("\n## Per-dimension breakdown\n")
    out.append(table)
    return "\n".join(out)


def analyze_fourth() -> str | None:
    sc_path = PILOT_DIR / "fourth_judge_scores.jsonl"
    if not sc_path.exists():
        return None
    pairs: list[tuple[int, int]] = []
    by_dim: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_model: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_condition: dict[str, list[tuple[int, int]]] = defaultdict(list)
    skipped = 0
    rows_seen = set()
    for line in sc_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("item_id") in rows_seen:
            continue
        rows_seen.add(d["item_id"])
        cs = d.get("claude_score")
        pm = d.get("panel_median")
        if not isinstance(cs, int) or cs < 0 or cs > 3:
            skipped += 1
            continue
        if not isinstance(pm, int) or pm < 0 or pm > 3:
            skipped += 1
            continue
        pairs.append((cs, pm))
        by_dim[d["dimension"]].append((cs, pm))
        by_model[d["model"]].append((cs, pm))
        by_condition[d["condition"]].append((cs, pm))

    overall = compute_metrics(pairs)
    rows_dim = _per_dimension(by_dim)
    table_dim = _format_table(
        rows_dim,
        [
            ("dimension", "Dimension", ""),
            ("n", "N", ""),
            ("exact", "Exact %", ".1%"),
            ("adjacent", "Adjacent %", ".1%"),
            ("mad", "MAD", ".2f"),
            ("qw_kappa", "QW-κ", ".3f"),
        ],
    )
    rows_model = []
    for model, p in sorted(by_model.items()):
        rows_model.append({"model": model, **compute_metrics(p)})
    table_model = _format_table(
        rows_model,
        [
            ("model", "Model", ""),
            ("n", "N", ""),
            ("exact", "Exact %", ".1%"),
            ("adjacent", "Adjacent %", ".1%"),
            ("mad", "MAD", ".2f"),
            ("qw_kappa", "QW-κ", ".3f"),
        ],
    )
    rows_cond = []
    for cond, p in sorted(by_condition.items()):
        rows_cond.append({"condition": cond, **compute_metrics(p)})
    table_cond = _format_table(
        rows_cond,
        [
            ("condition", "Condition", ""),
            ("n", "N", ""),
            ("exact", "Exact %", ".1%"),
            ("adjacent", "Adjacent %", ".1%"),
            ("mad", "MAD", ".2f"),
            ("qw_kappa", "QW-κ", ".3f"),
        ],
    )

    out = []
    out.append("# Claude-as-Fourth-Judge Agreement (vs existing 3-judge panel median)\n")
    out.append(
        "Methodology: Claude Sonnet 4.6 was given the same judge system prompt, "
        "rubric, and few-shot examples as the existing panel "
        "(Devstral-2 123B, GPT-oss 120B, Gemma 4 31B), then scored a stratified "
        "N=500 sample of Layer-1 instances. This is *not* human validation; it is an "
        "inter-LLM-judge consistency check across model families.\n"
    )
    out.append(f"Items scored: **{overall['n']}**  (skipped {skipped})\n")
    out.append("\n## Headline numbers\n")
    out.append(f"- Exact agreement: **{overall['exact']:.1%}**")
    out.append(f"- Adjacent agreement (±1): **{overall['adjacent']:.1%}**")
    out.append(f"- Mean absolute deviation: **{overall['mad']:.2f}**")
    out.append(f"- Quadratic-weighted Cohen's κ: **{overall['qw_kappa']:.3f}**")
    out.append("\n## Per-dimension\n")
    out.append(table_dim)
    out.append("\n\n## Per-model\n")
    out.append(table_model)
    out.append("\n\n## Per-condition\n")
    out.append(table_cond)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["human", "fourth", "both"], default="both", nargs="?")
    args = ap.parse_args()

    if args.which in ("human", "both"):
        h = analyze_human()
        if h is None:
            print("[human] pilot_scores.csv or pilot_ground_truth.json missing; skipping.")
        else:
            out_path = PILOT_DIR / "pilot_results.md"
            out_path.write_text(h + "\n")
            print(f"[human] wrote {out_path}")
            print(h)
            print()

    if args.which in ("fourth", "both"):
        f = analyze_fourth()
        if f is None:
            print("[fourth] fourth_judge_scores.jsonl missing; skipping.")
        else:
            out_path = PILOT_DIR / "fourth_judge_results.md"
            out_path.write_text(f + "\n")
            print(f"[fourth] wrote {out_path}")
            print(f)


if __name__ == "__main__":
    main()
