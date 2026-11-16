"""Check paper numbers against preserved evidence (B-line final paper).

Path adaptation: the pure-code anonymous repo has no paper/ or results/
evidence; those tests skip. In the full public package (paper/ and
evidence/results/ at the package root, plus code/validation/), all run.

Reconciliation targets (final paper, "RAWPrism: A Three-Task Dataset and a
Unified Model for RAW-Domain Image Fusion"):
  - tab:mff      : legacy MFF main table; per_scene_mff.json (v10; the final
                   checkpoint keeps the v10 MFF branch, so its row repeats
                   these values)
  - tab:ablation : v10 joint row vs per_scene_mef.json / per_scene_mff.json
  - tab:ivif     : Protocol-C IVIF main table vs the 280-scene replay csv
  - tab:mffcorrected (supplement) vs mff_c_final/summary.json
"""
import csv
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = next(
    (p for p in (ROOT / "paper", ROOT.parent / "paper") if p.is_dir()),
    None,
)
RESULTS = next(
    (p for p in (ROOT / "results", ROOT.parent / "evidence/results") if p.is_dir()),
    None,
)
IVIF_CSV = next(
    (p for p in (ROOT / "validation/ivif_c_repro_lsy.csv",
                 ROOT.parent / "code/validation/ivif_c_repro_lsy.csv") if p.is_file()),
    None,
)
NEED_PAPER = PAPER is not None and (PAPER / "RAWPrism_final.tex").is_file()

pytestmark = pytest.mark.skipif(not NEED_PAPER or RESULTS is None,
                                reason="no paper/results evidence in this repo layout")

if NEED_PAPER:
    MAIN = (PAPER / "RAWPrism_final.tex").read_text()
    SUPP = (PAPER / "RAWPrism_supplement.tex").read_text()


def table_rows(text, label):
    block = text.split("\\label{" + label + "}", 1)[1].split("\\end{table", 1)[0]
    rows = {}
    for line in block.splitlines():
        cells = line.strip().split("&")
        if len(cells) < 2 or not line.rstrip().endswith("\\\\"):
            continue
        values = [re.search(r"[-+]?\d+\.\d+", cell) for cell in cells[1:]]
        if values and all(value is not None for value in values):
            rows[cells[0].strip()] = [float(value.group()) for value in values]
    return rows


def named_row(rows, name):
    return next(value for key, value in rows.items() if name in key)


def test_legacy_mff_main_means_remain_legacy():
    rows = table_rows(MAIN, "tab:mff")
    results = json.loads((RESULTS / "per_scene_mff.json").read_text())
    for printed, method in (("RAWPrism", "rawprism_v10"), ("SwinFusion", "swinfusion"),
                            ("UniFusion", "unifusion"), ("TC-MoA", "tcmoa")):
        scenes = [row for row in results if row["method"] == method]
        assert len(scenes) == 15
        for index, metric in ((3, "MI"), (4, "SCD")):
            mean = sum(row[metric] for row in scenes) / len(scenes)
            assert named_row(rows, printed)[index] == pytest.approx(round(mean, 3))
    assert named_row(rows, "RAWPrism")[3] == 5.758


def test_corrected_supplement_has_all_methods_and_real_rawprism_mean():
    rows = table_rows(SUPP, "tab:mffcorrected")
    assert set(rows) == {"UniFusion", "SwinFusion", "CDDFuse", "IFCNN", "Laplacian", "RAWPrism", "TC-MoA"}
    result = json.loads((RESULTS / "mff_c_final/summary.json").read_text())["summary"]["rawfinal"]["mff"]
    metrics = ["EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "SSIM"]
    assert rows["RAWPrism"] == pytest.approx([round(result[key], 3) for key in metrics])
    assert rows["TC-MoA"][4] < 0
    assert rows["TC-MoA"][3:5] == [2.194, -0.483]


def test_v10_joint_ablation_matches_per_scene_evidence():
    rows = table_rows(MAIN, "tab:ablation")
    v10 = named_row(rows, "Full (v10 joint)")
    # column layout: IVIF MI/SCD/Q, MEF MI/SCD/Q, MFF MI/SCD/Q
    assert v10[0] == 2.477
    mef = json.loads((RESULTS / "per_scene_mef.json").read_text())
    mef_scenes = [row for row in mef if row["method"] == "rawprism_v10"]
    mef_mi = sum(row["MI"] for row in mef_scenes) / len(mef_scenes)
    mef_scd = sum(row["SCD"] for row in mef_scenes) / len(mef_scenes)
    assert v10[3] == pytest.approx(round(mef_mi, 3)) == 7.196
    assert v10[4] == pytest.approx(round(mef_scd, 3)) == 0.381
    mff = json.loads((RESULTS / "per_scene_mff.json").read_text())
    mff_scenes = [row for row in mff if row["method"] == "rawprism_v10"]
    mff_mi = sum(row["MI"] for row in mff_scenes) / len(mff_scenes)
    mff_scd = sum(row["SCD"] for row in mff_scenes) / len(mff_scenes)
    assert v10[6] == pytest.approx(round(mff_mi, 3)) == 5.758
    assert v10[7] == pytest.approx(round(mff_scd, 3)) == 0.424


def test_ivif_c_table_matches_replay_csv():
    if IVIF_CSV is None:
        pytest.skip("Protocol-C replay csv not present in this layout")
    metric_names = ["EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "CC", "SSIM", "PSNR"]
    with IVIF_CSV.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["scene"] for row in rows}) == 280
    printed = named_row(table_rows(MAIN, "tab:ivif"), "RAWPrism (ours)")
    for metric, value in zip(metric_names, printed):
        mean = sum(float(row[metric]) for row in rows) / len(rows)
        assert value == pytest.approx(round(mean, 3)), (metric, value, mean)


def test_splits_and_rounding_match_released_evidence():
    mef = json.loads((RESULTS / "per_scene_mef.json").read_text())
    scenes = [row for row in mef if row["method"] == "rawprism_v10"]
    split = json.loads((ROOT / "splits/mef_scene_split.json").read_text())["splits"]
    ids = {row["scene"] for row in scenes}
    assert [len(ids & set(split[key])) for key in ("train", "val", "test")] == [0, 0, 42]
    assert "338/42/42" in MAIN and "338/42/42" in SUPP


def test_llm_before_references_and_source_assets_present():
    assert MAIN.index("\\section*{LLM Usage Statement}") < MAIN.index("\\bibliography{references}")
    for text in (MAIN, SUPP, (PAPER / "related_work.tex").read_text()):
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            assert (PAPER / name).is_file(), name
