"""Build the final single-checkpoint model from the v10 joint checkpoint and the
IVIF branch.

The deployed checkpoint uses:
  * all v10 weights for the shared backbone, RFT, MEF/MFF adapters, and heads;
  * the IVIF adapter and head from the v2 fine-tune;
  * visible residual scale 0.1 at inference.

Usage (full v2 checkpoint):
  python scripts/build_final_checkpoint.py \
      --v10 runs/v10/best.pt \
      --v2 runs/v10_iviffix_ivif/best.pt \
      --out checkpoints/rawprism_final_inference.pt

Usage (small 8 KB delta, recommended when the server is remote):
  python scripts/build_final_checkpoint.py \
      --v10 runs/v10/best.pt \
      --delta checkpoints/ivif_branch_delta.pt \
      --out checkpoints/rawprism_final_inference.pt
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10", required=True)
    parser.add_argument("--v2", default=None)
    parser.add_argument("--delta", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (args.v2 is None) == (args.delta is None):
        raise SystemExit("provide exactly one of --v2 or --delta")

    v10 = torch.load(args.v10, map_location="cpu", weights_only=False)
    state = {k: v.clone() for k, v in v10["model"].items()}
    if args.v2 is not None:
        source = torch.load(args.v2, map_location="cpu", weights_only=False)["model"]
        for key, value in source.items():
            if key.startswith(("adapters.ivif.", "heads.ivif.")):
                state[key] = value.clone()
    else:
        source = torch.load(args.delta, map_location="cpu", weights_only=False)
        for key, value in source.items():
            state[key] = value.clone()

    cfg = copy.deepcopy(v10["config"])
    cfg["tasks"] = ["ivif", "mef", "mff"]
    cfg["visible_residual"] = True
    cfg["visible_residual_scale"] = 0.1
    arch = copy.deepcopy(v10["architecture_config"])
    arch["tasks"] = ["ivif", "mef", "mff"]
    arch["visible_residual"] = True
    arch["visible_residual_scale"] = 0.1

    out = {
        "model": state,
        "architecture": v10.get("architecture"),
        "architecture_config": arch,
        "config": cfg,
        "schema_version": v10.get("schema_version", 1),
        "merge_note": "v10 all-task checkpoint + v2 IVIF adapter/head, alpha=0.1",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("wrote", args.out, "tensors", len(state))


if __name__ == "__main__":
    main()
