"""Verify that numbers printed in the paper match the analysis pipeline's output.

The paper is the last link in a chain that starts at `results/*.json`, runs through
`run_analysis.py` into `analysis_output/*.csv`, and ends in LaTeX typed by hand. Hand-typing
is where numbers drift, and a manual cross-check missed several drifts in the submitted
version. This test closes that gap mechanically.

Run directly (`python3 tests/test_paper_consistency.py`) or under pytest.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TEX = ROOT / "paper_overleaf" / "pro_worker_benchmark.tex"
OUT = ROOT / "analysis_output"


def _tex() -> str:
    return TEX.read_text(encoding="utf-8")


def check_pwi_table() -> list[str]:
    """Every row of the PWI table must match pwi_scores.csv, including the delta."""
    errors = []
    df = pd.read_csv(OUT / "pwi_scores.csv").pivot(
        index="Model", columns="Variant", values="PWI"
    )
    body = re.search(
        r"\\label\{tab:pwi_results\}.*?\\midrule(.*?)\\bottomrule", _tex(), re.S
    )
    if not body:
        return ["PWI results table not found in the .tex"]

    seen = set()
    for line in body.group(1).strip().split(r"\\"):
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4:
            continue
        model = cells[0]
        if model not in df.index:
            errors.append(f"PWI table lists unknown model {model!r}")
            continue
        seen.add(model)
        base, prompted = float(cells[1]), float(cells[2])
        delta = float(cells[3].replace("+", ""))
        exp_b, exp_p = df.loc[model, "baseline"], df.loc[model, "with_system_prompt"]
        if abs(base - exp_b) > 0.05:
            errors.append(f"{model}: baseline PWI {base} in paper, {exp_b} in CSV")
        if abs(prompted - exp_p) > 0.05:
            errors.append(f"{model}: prompted PWI {prompted} in paper, {exp_p} in CSV")
        if abs(delta - (prompted - base)) > 0.05:
            errors.append(
                f"{model}: delta {delta} does not equal {prompted} - {base}"
            )
    missing = set(df.index) - seen
    if missing:
        errors.append(f"PWI table is missing models: {sorted(missing)}")
    return errors


def check_abstract_ranges() -> list[str]:
    """The ranges quoted in the abstract must bracket the actual PWI values."""
    errors = []
    df = pd.read_csv(OUT / "pwi_scores.csv").pivot(
        index="Model", columns="Variant", values="PWI"
    )
    base, prompted = df["baseline"], df["with_system_prompt"]
    delta = prompted - base
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", _tex(), re.S)
    if not abstract:
        return ["abstract not found"]
    text = abstract.group(1)

    expectations = [
        (r"Baseline PWI sits between ([\d.]+) and ([\d.]+)", base.min(), base.max()),
        (r"raises PWI to ([\d.]+)-([\d.]+)", prompted.min(), prompted.max()),
        (r"\$\\Delta\$ = \+([\d.]+) to \+([\d.]+)", delta.min(), delta.max()),
    ]
    for pattern, lo_exp, hi_exp in expectations:
        m = re.search(pattern, text)
        if not m:
            errors.append(f"abstract pattern not found: {pattern}")
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        if abs(lo - lo_exp) > 0.05 or abs(hi - hi_exp) > 0.05:
            errors.append(
                f"abstract says {lo}-{hi} for {pattern!r}; data gives "
                f"{lo_exp:.1f}-{hi_exp:.1f}"
            )
    return errors


def check_irr_table() -> list[str]:
    """Every reliability figure in the appendix must match irr_table.csv."""
    errors = []
    path = OUT / "irr_table.csv"
    if not path.exists():
        return ["irr_table.csv missing; run run_analysis.py"]
    df = pd.read_csv(path).set_index("Dimension")
    body = re.search(r"\\label\{tab:irr\}.*?\\midrule(.*?)\\bottomrule", _tex(), re.S)
    if not body:
        return ["IRR table not found in the .tex"]

    alias = {
        "Complementarity (CO)": "complementarity",
        "Anti-Sycophancy (AS)": "anti_sycophancy",
        "Cognitive Forcing (CF)": "cognitive_forcing",
        "Draft Annotation (DA)": "draft_annotation",
        "Ethical Surfacing (ES)": "ethical_surfacing",
        "Contrastive Explanation (CE)": "contrastive_explanation",
        "Skill Preservation (SP)": "skill_preservation",
        "Metacognitive Calibration (MC)": "metacognitive_calibration",
        "Uncertainty Transparency (UT)": "uncertainty_transparency",
        "Appropriate Reliance (AR\\_d)": "appropriate_reliance",
        "Pooled": "POOLED",
    }
    cols = ["N", "exact", "adjacent", "qwk", "alpha", "spread"]

    for line in body.group(1).replace(r"\midrule", "").strip().split(r"\\"):
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 7:
            continue
        dim = alias.get(cells[0])
        if dim is None:
            errors.append(f"IRR table has unrecognised row label {cells[0]!r}")
            continue
        for col, cell in zip(cols, cells[1:]):
            try:
                got = float(cell)
            except ValueError:
                errors.append(f"{dim}/{col}: cannot parse {cell!r}")
                continue
            # The paper prints a rounded value, so the correct comparison rounds the
            # source datum to the same precision rather than allowing a fixed tolerance.
            dp = len(cell.split(".")[1]) if "." in cell else 0
            exp = round(float(df.loc[dim, col]), dp)
            if got != exp:
                errors.append(f"{dim}/{col}: paper {got}, data rounds to {exp}")
    return errors


def check_weights_match_config() -> list[str]:
    """Weights printed in Table 1 must match config.yaml."""
    sys.path.insert(0, str(ROOT / "src"))
    import analysis  # noqa: E402

    errors = []
    weights = analysis.load_weights()
    abbrev = {
        "Cognitive Forcing": "cognitive_forcing",
        "Contrastive Explanation": "contrastive_explanation",
        "Skill Preservation": "skill_preservation",
        "Draft Annotation": "draft_annotation",
        "Uncertainty Transparency": "uncertainty_transparency",
        "Complementarity": "complementarity",
        "Anti-Sycophancy": "anti_sycophancy",
        "Metacognitive Calibration": "metacognitive_calibration",
        "Ethical Surfacing": "ethical_surfacing",
    }
    body = re.search(
        r"\\label\{tab:dimensions\}.*?\\midrule(.*?)\\bottomrule", _tex(), re.S
    )
    if not body:
        return ["dimensions table not found in the .tex"]
    for line in body.group(1).strip().split(r"\\"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 5:
            continue
        key = abbrev.get(cells[0])
        if key is None:
            continue
        # A dash-leading cell marks a dimension excluded from the composite; load_weights()
        # drops zero-weight dimensions, so the dimension must be absent there.
        if cells[4].lstrip().startswith("---"):
            if key in weights:
                errors.append(f"{key}: paper marks it excluded but config.yaml weights it")
            continue
        m = re.match(r"([\d.]+)", cells[4])
        if not m:
            errors.append(f"{key}: cannot parse weight cell {cells[4]!r}")
            continue
        got = float(m.group(1))
        exp = weights.get(key)
        if exp is None:
            errors.append(f"{key}: has a weight in the paper but not in config.yaml")
        elif abs(got - exp) > 1e-9:
            errors.append(f"{key}: paper weight {got}, config.yaml {exp}")
    return errors


CHECKS = {
    "PWI table vs pwi_scores.csv": check_pwi_table,
    "abstract ranges vs pwi_scores.csv": check_abstract_ranges,
    "IRR table vs irr_table.csv": check_irr_table,
    "Table 1 weights vs config.yaml": check_weights_match_config,
}


def test_paper_consistency():
    failures = {name: errs for name, fn in CHECKS.items() if (errs := fn())}
    assert not failures, "\n".join(
        f"[{name}] {e}" for name, errs in failures.items() for e in errs
    )


def main() -> int:
    total = 0
    for name, fn in CHECKS.items():
        errs = fn()
        total += len(errs)
        print(f"{'FAIL' if errs else 'ok  '}  {name}")
        for e in errs:
            print(f"        {e}")
    print(f"\n{total} inconsistencies")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
