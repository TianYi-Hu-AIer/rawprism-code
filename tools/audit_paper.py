"""Check the main-table rows against the released result files.

This is a focused audit: it verifies the RAWPrism and Laplacian rows, not every
baseline. It exits 0 only when all checked numbers agree within 0.0015.

Usage: python tools/audit_paper.py [--paper paper/RAWPrism_final.tex]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"[-+]?\d*\.\d+")
TOKENS = {
    "tab:ivif": ["EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "CC", "SSIM", "PSNR"],
    "tab:mef": ["EN", "SD", "Q_ABF", "MI", "SCD", "MEFSSIM", "VIF", "SSIM"],
    "tab:mff": ["EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "SSIM"],
}


def table_rows(tex: str, label: str) -> dict[str, list[float]]:
    block = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{table\*\}", tex, re.S)
    if not block:
        raise SystemExit(f"table {label} not found")
    rows: dict[str, list[float]] = {}
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        key = re.sub(r"\\[a-zA-Z]+|[{}$]", "", cells[0]).strip()
        values = []
        for cell in cells[1:]:
            match = NUM.search(cell.replace("$", ""))
            if match:
                values.append(float(match.group(0)))
        if values:
            rows[key] = values
    return rows


def check(name: str, got: list[float], expected: list[float], issues: list[str]) -> None:
    if len(got) != len(expected):
        issues.append(f"{name}: {len(got)} values vs {len(expected)} expected")
        return
    for metric, a, b in zip(TOKENS_CURRENT, got, expected):
        if abs(a - b) > 0.0015:
            issues.append(f"{name} {metric}: paper {a} vs result {b:.4f}")


def mean_metric(path: Path, method: str, metric: str) -> float:
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [r for r in rows if r["method"] == method]
    return sum(float(r[metric]) for r in selected) / len(selected)


TOKENS_CURRENT: list[str] = []


def main() -> int:
    global TOKENS_CURRENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=str(ROOT / "paper/RAWPrism_final.tex"))
    args = parser.parse_args()
    tex = Path(args.paper).read_text(encoding="utf-8")
    issues: list[str] = []

    # IVIF final
    summary = json.loads((ROOT / "results/ivif_c_final/summary.json").read_text(encoding="utf-8"))
    ivif = summary["summary"]["head_a10"]["ivif"]
    TOKENS_CURRENT = TOKENS["tab:ivif"]
    got = table_rows(tex, "tab:ivif")
    row = next((v for k, v in got.items() if "RAWPrism" in k), None)
    if row is None:
        issues.append("IVIF: RAWPrism row not found")
    else:
        check("IVIF RAWPrism", row, [ivif[c] for c in TOKENS_CURRENT], issues)
    lap = json.loads((ROOT / "results/ivif_c_lap/summary.json").read_text(encoding="utf-8"))["summary"]["lap_c"]["ivif"]
    row = next((v for k, v in got.items() if "Laplacian" in k), None)
    if row is None:
        issues.append("IVIF: Laplacian row not found")
    else:
        check("IVIF Laplacian", row, [lap[c] for c in TOKENS_CURRENT], issues)

    # MEF/MFF: only Q_ABF/MI/SCD are present in per_scene files
    for label, path, cols in (
        ("tab:mef", ROOT / "results/per_scene_mef.json", TOKENS["tab:mef"]),
        ("tab:mff", ROOT / "results/per_scene_mff.json", TOKENS["tab:mff"]),
    ):
        row = next((v for k, v in table_rows(tex, label).items() if "RAWPrism" in k), None)
        if row is None:
            issues.append(f"{label}: RAWPrism row not found")
            continue
        for metric in ("Q_ABF", "MI", "SCD"):
            sample = json.loads(path.read_text(encoding="utf-8"))
            if metric not in sample[0]:
                continue
            idx = cols.index(metric)
            expected = mean_metric(path, "rawprism_v10", metric)
            if abs(row[idx] - expected) > 0.0015:
                issues.append(f"{label} {metric}: paper {row[idx]} vs result {expected:.4f}")

    if issues:
        print("MISMATCH")
        for issue in issues:
            print(" -", issue)
        return 1
    print("OK: checked rows match the released result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
