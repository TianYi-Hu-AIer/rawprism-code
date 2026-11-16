"""Protocol-enforcing re-evaluation (metrics v2) for every method x task.

Fixes the three evaluator defects the review lists under W1/W2:

1. Metrics are the validated v2 definitions (Q_ABF uses the fused gradient,
   SCD is the cross-difference Pearson form, VIFp pools information across
   scales).
2. Missing predictions and shape mismatches are errors, not silent ``continue``:
   every method must evaluate exactly the pre-declared test scenes.
3. Field of view is explicit.  ``--fov strict`` (default) requires identical
   shapes; ``--fov resize`` resizes the *sources* to the fused full view and
   records the flag, instead of comparing a top-left crop with a downscaled
   full image.

Outputs
-------
* ``per_image.csv``   one row per (method, task, scene) - required for paired CIs
* ``summary.json``    means/medians/n per method x task
* ``summary.md``      markdown table
* ``paired.json``     scene-level paired differences vs a reference method

Usage
-----
  python tools/evaluate_v2.py --preds preds_v2 --data . --out metrics_v2 \
      --splits mef=<path> --splits mff=<path>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_v2 as M  # noqa: E402

TASKS = ("ivif", "mef", "mff")
MAX_SIDE = 1024
CSV_FIELDS = ["method", "task", "scene", "fov_mode", "h", "w", "h_source", "w_source",
              "EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "CC", "SSIM", "PSNR", "SF", "AG"]


# ------------------------------------------------------------------ I/O helpers
def read_rgb(path: Path) -> np.ndarray:
    # np.fromfile + imdecode is used first: cv2.imread returns None on non-ASCII
    # Windows paths (and prints a decoder warning for every such file).
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    except OSError:
        image = None
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image[..., ::-1].astype(np.float32) / 255.0


def resize_max_side(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def resize_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------- protocol
def mef_scenes(split_path: Path) -> list[str]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return sorted(payload["splits"]["test"])


def mff_scenes(split_path: Path) -> list[str]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    if "test" in splits:
        return sorted(splits["test"])
    raise KeyError("MFF split has no explicit 'test'; refusing to fall back to validation")


def ivif_scenes(data_root: Path) -> list[str]:
    root = data_root / "ivif" / "test"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def scene_list(task: str, data_root: Path, splits: dict[str, Path]) -> list[str]:
    if task == "ivif":
        return ivif_scenes(data_root)
    if task not in splits:
        raise KeyError("task {} needs --splits {}=".format(task, task))
    return mef_scenes(splits[task]) if task == "mef" else mff_scenes(splits[task])


def source_paths(task: str, data_root: Path, scene: str) -> tuple[Path, Path]:
    if task == "ivif":
        base = data_root / "ivif" / "test" / scene
        return base / "vis_rgb.png", base / "ir.png"
    if task == "mef":
        base = data_root / "rgb" / "mef" / scene
        return base / "under.png", base / "over.png"
    base = data_root / "rgb" / "mff" / scene
    return base / "far.png", base / "near.png"


# ---------------------------------------------------------------- metric extras
def extras(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """VIF / CC / SSIM / PSNR / SF / AG on a common 1024 max-side luminance grid.

    VIF is deliberately computed here (not at native 12 MP resolution): the
    four-scale local statistics cost ~40 s per image at 12 MP, and the released
    protocol already declared a 1024 max-side grid for VIF/SSIM/PSNR.
    """
    a, b, f = resize_max_side(a), resize_max_side(b), resize_max_side(f)
    ga, gb, gf = M.to_gray(a), M.to_gray(b), M.to_gray(f)

    def _cc(x, y):
        x = x - x.mean()
        y = y - y.mean()
        den = float(np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel()))
        return 0.0 if den <= M.EPS else float(np.sum(x * y) / den)

    def _sf(g):
        rf = np.sqrt(np.mean((g[:, 1:] - g[:, :-1]) ** 2))
        cf = np.sqrt(np.mean((g[1:, :] - g[:-1, :]) ** 2))
        return float(np.sqrt(rf * rf + cf * cf))

    def _ag(g):
        dx = g[:-1, 1:] - g[:-1, :-1]
        dy = g[1:, :-1] - g[:-1, :-1]
        return float(np.mean(np.sqrt((dx * dx + dy * dy) / 2.0)))

    return {
        "VIF": M.vif_fusion(a, b, f),
        "CC": 0.5 * (_cc(ga, gf) + _cc(gb, gf)),
        "SSIM": 0.5 * (M.ssim(a, f) + M.ssim(b, f)),
        "PSNR": 0.5 * (M.psnr(a, f) + M.psnr(b, f)),
        "SF": _sf(gf),
        "AG": _ag(gf),
    }


# ------------------------------------------------------------------- evaluation
def evaluate(args: argparse.Namespace) -> dict:
    data_root = Path(args.data).resolve()
    pred_root = Path(args.preds).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {k: Path(v).resolve() for k, v in args.splits}
    tasks = tuple(args.tasks or TASKS)
    methods = args.methods or sorted(p.name for p in pred_root.iterdir() if p.is_dir())
    rows: list[dict] = []
    missing: dict[str, list[str]] = {}
    shape_errors: list[dict] = []

    for task in tasks:
        scenes = scene_list(task, data_root, splits)
        print("[{}] declared test scenes: {}".format(task, len(scenes)))
        for method in methods:
            got = 0
            for scene in scenes:
                fused_path = pred_root / method / task / (scene + ".png")
                if not fused_path.is_file():
                    missing.setdefault("{}/{}".format(method, task), []).append(scene)
                    continue
                sa_path, sb_path = source_paths(task, data_root, scene)
                sa, sb, fused = read_rgb(sa_path), read_rgb(sb_path), read_rgb(fused_path)
                fov_mode = "strict"
                if sa.shape[:2] != fused.shape[:2] or sb.shape[:2] != fused.shape[:2]:
                    if args.fov == "resize":
                        sa = resize_to(sa, *fused.shape[:2])
                        sb = resize_to(sb, *fused.shape[:2])
                        fov_mode = "resize"
                    else:
                        shape_errors.append({"method": method, "task": task, "scene": scene,
                                             "fused": list(fused.shape[:2]),
                                             "source_a": list(sa.shape[:2]),
                                             "source_b": list(sb.shape[:2])})
                        continue
                if args.grid:
                    sa = resize_max_side(sa, args.grid)
                    sb = resize_max_side(sb, args.grid)
                    fused = resize_max_side(fused, args.grid)
                    fov_mode = fov_mode + "+grid{}".format(args.grid)
                row = {"method": method, "task": task, "scene": scene, "fov_mode": fov_mode,
                       "h": fused.shape[0], "w": fused.shape[1],
                       "h_source": sa.shape[0], "w_source": sa.shape[1]}
                row.update(M.evaluate_pair(sa, sb, fused))
                row.update(extras(sa, sb, fused))
                rows.append(row)
                got += 1
            print("    {:<18} evaluated {}/{}".format(method, got, len(scenes)))

    if shape_errors and not args.allow_shape_mismatch:
        (out_dir / "shape_errors.json").write_text(json.dumps(shape_errors, indent=2), encoding="utf-8")
        raise SystemExit("{} shape mismatches (use --fov resize to declare a full-view resize, "
                         "or --allow-shape-mismatch to record and skip)".format(len(shape_errors)))
    if missing and not args.allow_missing:
        (out_dir / "missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
        raise SystemExit("missing predictions for {} method/task pairs; refusing to average "
                         "different test subsets (see missing.json)".format(len(missing)))

    # ------------------------------------------------------------------ outputs
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})

    metrics = ["EN", "SD", "Q_ABF", "MI", "SCD", "VIF", "CC", "SSIM", "PSNR", "SF", "AG"]
    summary: dict[str, dict[str, dict]] = {}
    for method in methods:
        for task in tasks:
            group = [r for r in rows if r["method"] == method and r["task"] == task]
            if not group:
                continue
            summary.setdefault(method, {})[task] = {
                "n": len(group),
                "fov_modes": sorted({r["fov_mode"] for r in group}),
                **{m: float(np.mean([r[m] for r in group])) for m in metrics},
                **{"{}_median".format(m): float(np.median([r[m] for r in group])) for m in metrics},
            }

    paired = {}
    if args.reference:
        for task in tasks:
            for method in methods:
                if method == args.reference:
                    continue
                base = {r["scene"]: r for r in rows if r["method"] == args.reference and r["task"] == task}
                other = {r["scene"]: r for r in rows if r["method"] == method and r["task"] == task}
                common = sorted(set(base) & set(other))
                if not common:
                    continue
                entry = {"n_paired": len(common)}
                for m in metrics:
                    diff = np.array([other[s][m] - base[s][m] for s in common], dtype=np.float64)
                    lo, hi = _bootstrap_ci(diff, seed=20261031)
                    entry[m] = {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                "win": int((diff > 0).sum()), "tie": int((diff == 0).sum()),
                                "loss": int((diff < 0).sum())}
                paired["{}/{}".format(method, task)] = entry

    (out_dir / "summary.json").write_text(
        json.dumps({"metric_version": M.METRIC_VERSION, "reference": args.reference,
                    "methods": methods, "summary": summary, "paired": paired}, indent=2),
        encoding="utf-8")
    (out_dir / "summary.md").write_text(_markdown(summary, metrics, tasks), encoding="utf-8")
    if paired:
        (out_dir / "paired.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print("wrote", out_dir)
    return summary


def _bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 0):
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def _markdown(summary, metrics, tasks) -> str:
    lines = []
    for task in tasks:
        lines.append("## {}".format(task.upper()))
        lines.append("| Method | n | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---|" * (len(metrics) + 1))
        for method, per_task in summary.items():
            if task not in per_task:
                continue
            d = per_task[task]
            lines.append("| {} | {} | ".format(method, d["n"]) +
                         " | ".join("{:.3f}".format(d[m]) for m in metrics) + " |")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", required=True, help="root containing <method>/<task>/<scene>.png")
    parser.add_argument("--data", required=True, help="root containing ivif/, rgb/mef/, rgb/mff/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=None,
                        help="tasks to evaluate (default: all); use ivif to skip MEF/MFF")
    parser.add_argument("--splits", nargs="*", default=[], help="task=path/to/split.json")
    parser.add_argument("--fov", choices=["strict", "resize"], default="strict")
    parser.add_argument("--grid", type=int, default=0,
                        help="resample sources and fused to this max side before metrics "
                             "(0 = native). A common grid removes resolution/FOV confounds.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-shape-mismatch", action="store_true")
    parser.add_argument("--reference", default=None, help="method name for paired scene-level CIs")
    parsed = parser.parse_args()
    parsed.splits = [tuple(item.split("=", 1)) for item in parsed.splits]
    return parsed


if __name__ == "__main__":
    evaluate(parse_args())
