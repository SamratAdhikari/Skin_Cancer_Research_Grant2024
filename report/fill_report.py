"""
fill_report.py
--------------
Substitute ``{{dotted.path|fmt}}`` placeholders in ``report/report.md`` with
values from ``report/results.json`` (produced by notebook 06), writing
``report/report_final.md``.

This keeps the paper's numbers in lock-step with the evaluation code: run the
notebook, run this script, and the report is populated with zero manual
transcription. Any placeholder whose value is missing is left as a visible
``<TODO: path>`` marker and counted in the summary, so nothing is silently blank.

Usage:
    python report/fill_report.py                 # uses report/results.json
    python report/fill_report.py path/to/results.json report/out.md

Formatters (after the ``|``):
    pct   value*100 with 1 decimal + '%'     (0.8241 -> '82.4%')
    pct2  value*100 with 2 decimals + '%'
    f2    2 decimals                         (0.83   -> '0.83')
    f3    3 decimals                         (0.9123 -> '0.912')
    int   integer                            (12.0   -> '12')
    (none) str(value)
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = re.compile(r"\{\{([^}|]+?)(?:\|([a-z0-9]+))?\}\}")


def _resolve(path: str, data: dict):
    """Walk a dotted path through nested dicts; return None if any key missing."""
    cur = data
    for key in path.strip().split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _format(value, fmt: str | None) -> str:
    if value is None:
        return None  # signal "missing" to caller
    try:
        if fmt == "pct":
            return f"{float(value) * 100:.1f}%"
        if fmt == "pct2":
            return f"{float(value) * 100:.2f}%"
        if fmt == "f2":
            return f"{float(value):.2f}"
        if fmt == "f3":
            return f"{float(value):.3f}"
        if fmt == "int":
            return f"{int(round(float(value)))}"
    except (TypeError, ValueError):
        return None
    return str(value)


def fill(template: str, data: dict) -> tuple[str, int, list[str]]:
    """Return (filled_text, n_filled, missing_paths)."""
    n_filled = 0
    missing: list[str] = []

    def _sub(m: re.Match) -> str:
        nonlocal n_filled
        path, fmt = m.group(1), m.group(2)
        rendered = _format(_resolve(path, data), fmt)
        if rendered is None:
            missing.append(path.strip())
            return f"<TODO: {path.strip()}>"
        n_filled += 1
        return rendered

    return PLACEHOLDER.sub(_sub, template), n_filled, missing


def main(argv: list[str]) -> int:
    results_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "results.json")
    out_path = argv[2] if len(argv) > 2 else os.path.join(HERE, "report_final.md")
    template_path = os.path.join(HERE, "report.md")

    if not os.path.exists(results_path):
        print(f"[fill_report] results file not found: {results_path}")
        print("              run notebooks/06_evaluation_figures_gradcam.ipynb first.")
        return 1

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    filled, n_filled, missing = fill(template, data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(filled)

    print(f"[fill_report] wrote {out_path}")
    print(f"[fill_report] {n_filled} placeholders filled, {len(missing)} missing")
    if missing:
        uniq = sorted(set(missing))
        print("[fill_report] missing (left as <TODO: ...>):")
        for p in uniq:
            print(f"    - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
