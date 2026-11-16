# FinalFusion

## Code layout

The package keeps shared model code at the package root and isolates the three
task-specific input adapters:

```text
finalfusion/
  common.py             # shared CFA transforms, backbone, output head
  model.py              # task routing and model assembly
  ivif/                 # IVIF adapter and dataset helpers
  mef/                  # MEF adapter and all MEF support code
  mff/                  # MFF adapter and all MFF support code
  data.py, losses.py    # shared training dispatch and task loss dispatch
```

`from finalfusion.model import UnifiedFinalNet` remains the stable public
construction path; moving adapters does not change checkpoint field names or
the task routing API.

One PyTorch checkpoint for IVIF, MEF, and MFF. The model contains three
task-specific input adapters, one shared joint-12 U2Fusion backbone, and three
small task-specific latent heads. Inference uses an explicit task route.

## Internal contract

```text
task RAW pair + metadata
  -> task adapter
  -> two canonical latent6 tensors
  -> zero-initialized task 12-to-12 residual projection
  -> one shared 12-to-6 U2Fusion backbone
  -> zero-initialized task 6-to-6 residual head
  -> CFA synthesis and task output constraint
  -> task postprocessor
```

The adapters retain the current project behavior:

- IVIF packs/matches four-phase IR and zeros its two chroma latent planes.
- MEF maps native sources to the geometric-mean exposure domain and retains
  native/reference tensors for its relative-SNR training loss.
- MFF canonicalizes the NPZ CFA order, uses separate red/blue green selectors,
  samples fully valid registered crops, and restores the original packed order.

All three output heads use the same smooth sensor-range mapping with
`softplus(x, beta=50) - softplus(x-1, beta=50)`. MEF still exposes its
pre-clamp RAW for the range penalty, but no task uses a hard network clamp.

The shared backbone uses one initialization and optimizer in joint training.
This is a new multi-task architecture; it is not checkpoint-compatible with
the three original single-task directories.

The package is self-contained with respect to project code. `mef/support`
contains MEF discovery, cache, exposure, noise, reliability, chroma, and loss
logic; `mff/support` contains MFF registered-pair loading and RAW utilities;
IVIF-specific loading and loss logic live under `ivif`. No original task
directory or sibling Python package is imported at runtime.

## Loss v2

Each task uses at most five complementary terms and reports every term
separately in `history.jsonl`:

- IVIF: spatially adaptive intensity, dual-source structure, gradient
  magnitude, visible chroma, and flat-region artifact suppression.
- MEF: reliability-confidence Charbonnier, dual-exposure structure, reliable
  detail, chroma residual, and tolerant pre-clamp range regularization.
- MFF: normalized focus-weighted Charbonnier, focus-weighted structure,
  gradient magnitude, repeated-blur sharpness, and selected-source chroma.

MFF activity weights always sum to one and fall back to equal source weights
in flat regions. Loss weights are configured independently in each task's
`task_config` entry.

## Verify

From `FinalNet`:

```bash
python -m finalfusion.smoke_test
```

The data-free test runs all three forwards and losses, verifies that every task
backpropagates into the same backbone, and checks checkpoint round-tripping.

## Configure and train

Edit `finalfusion/config.example.json`, then run all three tasks:

```bash
python -m finalfusion.train --config finalfusion/config.example.json
```

Balanced joint training executes `steps_per_task` batches from each task per
epoch. A short loader is cycled, so dataset size alone cannot dominate the
shared backbone. `task_weights` scale task gradients and the combined
validation score.

The same entry point supports single-task ablations while retaining the same
architecture code:

```bash
python -m finalfusion.train \
  --config finalfusion/config.example.json \
  --tasks ivif
```

Outputs are `last.pt`, `best.pt`, `history.jsonl`, and `resolved_config.json`.
Resume with `--resume <checkpoint>` and the same task list.

### AutoDL one-command MEF run

With this project at `/root/autodl-tmp/finalfusion` and DatasetMEF at
`/root/autodl-tmp/DatasetMEF_denoised`, run:

```bash
bash /root/autodl-tmp/finalfusion/train_mef_autodl.sh
```

Upload the added `scene_0326` through `scene_0442` directories directly into
`/root/autodl-tmp/DatasetMEF_denoised`, alongside the existing scene
directories. Do not upload the unrelated `.accelerate` or other download
folders. The packaged split preserves all 305 previous assignments and adds
the 117 scenes with CFA-stratified allocation: 93 train, 12 validation, and 12
test (338/42/42 total).

The script validates the nested or flat DatasetMEF TIFF layout and requires
every discovered scene to occur exactly once in the packaged split. It then
creates or incrementally refreshes the packed cache and training-split noise
model, writes the training JSON, and starts single-task MEF training. If
`last.pt` exists, it resumes automatically. Set `FORCE_REBUILD_CACHE=1` to
regenerate the full cache instead of only adding missing scenes.

## Inference

IVIF input is one FMB-RAW scene directory:

```bash
python -m finalfusion.infer \
  --checkpoint D:/runs/finalfusion/best.pt \
  --task ivif \
  --input D:/data/FMB-RAW/test/scene0001 \
  --output D:/outputs/ivif
```

MEF input is a scene name. Data paths and black/white levels come from the
checkpoint config (or an explicitly supplied `--config`):

```bash
python -m finalfusion.infer \
  --checkpoint D:/runs/finalfusion/best.pt \
  --task mef \
  --input scene_0001 \
  --output D:/outputs/mef/scene_0001
```

MFF input and output are NPZ files:

```bash
python -m finalfusion.infer \
  --checkpoint D:/runs/finalfusion/best.pt \
  --task mff \
  --input D:/data/MFF/scene_0001/registered_raw_pair.npz \
  --output D:/outputs/mff/scene_0001_fused.npz
```

All routes use center-copy tiled inference without blending. MEF writes the
direct soft-clamped network RAW4 without source-chroma replacement. MFF writes
the fused result back in the input NPZ's original packed CFA order.

Every route exposes only its final fused result; there is no separate
pre/post-HSC network output. IVIF writes final RAW4/Bayer NPY files, MEF writes
final packed NPY and TIFF files, and MFF writes final `fused_packed` in its
metadata-preserving NPZ.

### AutoDL MEF test inference and ISP

After training, place the updated OpenCV ISP directory at
`/root/autodl-tmp/opencv_fixed_raw_compare_isp` and run:

```bash
bash /root/autodl-tmp/finalfusion/infer_mef_isp_autodl.sh
```

The default run loads `runs/finalfusion_mef/best.pt` once, infers the packaged
MEF test split, and writes direct network RAW outputs under
`outputs/finalfusion_mef_raw`. It then renders PNGs under
`outputs/finalfusion_mef_isp` using each scene's real DNG metadata. The ISP
enables OpenCV edge-aware demosaicing, fixed color-difference false-color
suppression, camera-domain highlight reconstruction, DNG color conversion,
and a ratio-preserving Reinhard highlight shoulder. RAW inputs default to
`DatasetMEF_denoised`, while per-scene DNG metadata defaults to the separate
`DatasetMEF_metadata_numeric` directory.

The defaults can be overridden without editing the script:

```bash
SPLIT=val FIXED_EV=0.5 TILE_SIZE=384 FORCE=1 \
  bash /root/autodl-tmp/finalfusion/infer_mef_isp_autodl.sh
```

`FORCE=0` resumes safely by skipping complete scene outputs. Set `CHECKPOINT`,
`CONFIG`, `METADATA_ROOT`, `RAW_OUTPUT_ROOT`, or `RGB_OUTPUT_ROOT` when the
dataset or training/output locations differ from the defaults.
