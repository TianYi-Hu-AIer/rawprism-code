# RAWPrism — Code Release (anonymous)

Code for the anonymous CVPR submission **"RAWPrism: A Three-Task Dataset and a
Unified Model for RAW-Domain Image Fusion"**: one 3.58M-parameter unified
fusion network covering multi-exposure (MEF), multi-focus (MFF), and
infrared–visible (IVIF) fusion in the RAW domain, before display rendering.

## Contents

```text
src/finalfusion/     model, training, inference, metrics (the Python package)
tools/               inference entry points and evaluation helpers
configs/v10.json     joint training / inference configuration
splits/              scene-level splits for the three tracks
scripts/             build_final_checkpoint.py — final checkpoint assembly
tests/               release test suite
validation/          checkpoint verification and replay records
requirements.txt
```

## Quickstart

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m finalfusion.smoke_test
```

The data-free smoke test runs all three task forwards and losses, verifies
backpropagation into the shared backbone, and checks checkpoint round-trips.
Run commands from the repository root with `src` on `PYTHONPATH` (the package
lives under `src/finalfusion`).

Final checkpoint assembly (the two required weights are in the restricted
data record; see the paper's availability statement):

```bash
python scripts/build_final_checkpoint.py \
    --v10 <v10_best.pt> --delta <ivif_branch_delta.pt> --out <final.pt>
```

Tensor-level verification of the merge is provided by
`validation/checkpoint_verify_lsy.json`.

## Verified reproduction scope

The release was independently reproduced end-to-end on a clean local machine
(Windows, RTX 5060 Laptop, PyTorch 2.8.0, rawpy 0.28) before publishing:

1. **Structure and tests**: `finalfusion.smoke_test` and the release test
   suite pass (paper-number reconciliation included: the IVIF table in the
   paper matches the archived 280-scene csv means exactly).
2. **Checkpoint rebuild**: assembling the final checkpoint from the released
   `v10_best.pt` + `ivif_branch_delta.pt` is **tensor-exact** (185 non-IVIF
   tensors equal v10 bit-for-bit; the 10 IVIF tensors equal the delta;
   3,584,408 parameters; all three task forwards finite).
3. **Metrics pipeline**: with the same self-generated Protocol-C reference
   (display-ISP rendering of the visible source), the 280-scene evaluation
   reproduces the paper's IVIF numbers to within the reference-version
   margin (EN 7.374 vs 7.392, MI 5.308 vs 5.284, CC 0.508 vs 0.501, SSIM
   0.717 vs 0.699, VIF 0.483 vs 0.495); the remaining shifts concentrate in
   PSNR/SCD, whose historical scoring references (`vis_rgb_rt.png` renders)
   are server-side artifacts not included in the public dataset. Regenerate
   references with the documented display ISP (code provided) and compare
   all methods under the same reference for like-for-like rankings.

Inference (per task, explicit task route):

```bash
PYTHONPATH=src python -m finalfusion.infer \
    --checkpoint <final.pt> --task ivif|mef|mff \
    --input <scene-dir|scene-name|pair.npz> --output <dir>
```

## Data

RAWPrism-Benchmark is available on Baidu Netdisk:

- [Download RAWPrism_Benchmark](https://pan.baidu.com/s/1OLz5bYI-DRutXGhjyGmSAA?pwd=zdpb)
- Extraction code: `zdpb`

Model weights are available in the
[v1.0-weights release](https://github.com/TianYi-Hu-AIer/rawprism-code/releases/tag/v1.0-weights).



## License

MIT. Anonymous during the review period.
