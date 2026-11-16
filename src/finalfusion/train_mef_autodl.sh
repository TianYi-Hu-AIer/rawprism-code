#!/usr/bin/env bash
set -Eeuo pipefail

# One-command AutoDL MEF preparation and training for FinalFusion.
# Override any path or setting by exporting the variable before running.
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
FINALFUSION_DIR="${FINALFUSION_DIR:-${AUTODL_ROOT}/finalfusion}"
DATA_ROOT="${DATA_ROOT:-${AUTODL_ROOT}/DatasetMEF_denoised}"
CACHE_ROOT="${CACHE_ROOT:-${AUTODL_ROOT}/cache/finalfusion_mef}"
RUN_DIR="${RUN_DIR:-${AUTODL_ROOT}/runs/finalfusion_mef}"
BLACK_LEVEL="${BLACK_LEVEL:-528}"
WHITE_LEVEL="${WHITE_LEVEL:-4095}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-0}"

EXPECTED_SAMPLE="${DATA_ROOT}/scene_0001/dng_components/over/raw/raw_unprocessed/scene_0001_raw_unprocessed_raw_full_frame.tiff"
PREPARE_MODULE="finalfusion.prepare_mef_cache"
SPLIT_JSON="${FINALFUSION_DIR}/mef_scene_split.json"
CONFIG_PATH="${RUN_DIR}/mef_train_config.json"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory not found: $1" >&2
    exit 1
  fi
}

require_dir "${FINALFUSION_DIR}"
require_dir "${DATA_ROOT}"
require_file "${FINALFUSION_DIR}/train.py"
require_file "${FINALFUSION_DIR}/model.py"
require_file "${FINALFUSION_DIR}/mef/support/raw.py"
require_file "${FINALFUSION_DIR}/mef/support/losses.py"
require_file "${EXPECTED_SAMPLE}"
require_file "${FINALFUSION_DIR}/prepare_mef_cache.py"
require_file "${SPLIT_JSON}"

mkdir -p "${CACHE_ROOT}" "${RUN_DIR}"
export PYTHONPATH="${AUTODL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[1/4] Validating DatasetMEF discovery and packaged split..."
python -c "from finalfusion.mef.support.raw import discover_pairs; from finalfusion.mef.support.data import split_pairs_from_json; pairs=discover_pairs(r'${DATA_ROOT}'); split=split_pairs_from_json(pairs, r'${SPLIT_JSON}'); print('MEF scenes:', len(pairs), {k: len(v) for k,v in split.items()})"

if [[ "${FORCE_REBUILD_CACHE}" == "1" ]] || ! python -m "${PREPARE_MODULE}" \
  --data-root "${DATA_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --black-level "${BLACK_LEVEL}" \
  --white-level "${WHITE_LEVEL}" \
  --check-only; then
  echo "[2/4] Preparing packed RAW cache and fitting the training-split noise model..."
  PREPARE_ARGS=(
    python -m "${PREPARE_MODULE}"
    --data-root "${DATA_ROOT}"
    --cache-root "${CACHE_ROOT}"
    --black-level "${BLACK_LEVEL}"
    --white-level "${WHITE_LEVEL}"
  )
  if [[ "${FORCE_REBUILD_CACHE}" == "1" ]]; then
    PREPARE_ARGS+=(--overwrite)
  fi
  "${PREPARE_ARGS[@]}"
else
  echo "[2/4] Reusing complete cache: ${CACHE_ROOT}"
fi

echo "[3/4] Writing resolved single-task MEF configuration..."
cat > "${CONFIG_PATH}" <<JSON
{
  "tasks": ["mef"],
  "output_dir": "${RUN_DIR}",
  "device": "cuda",
  "seed": 1337,
  "deterministic": false,
  "epochs": ${EPOCHS},
  "width": 44,
  "optimizer": "rmsprop",
  "learning_rate": 0.00003,
  "rmsprop_alpha": 0.6,
  "rmsprop_momentum": 0.15,
  "rmsprop_epsilon": 1e-8,
  "weight_decay": 0.0,
  "gradient_clip": 5.0,
  "amp": true,
  "amp_dtype": "bfloat16",
  "use_scheduler": true,
  "lr_patience": 4,
  "lr_factor": 0.3,
  "min_learning_rate": 0.000001,
  "task_weights": {"mef": 1.0},
  "task_config": {
    "mef": {
      "data_root": "${DATA_ROOT}",
      "cache_root": "${CACHE_ROOT}",
      "split_json": "${SPLIT_JSON}",
      "black_level": ${BLACK_LEVEL},
      "white_level": ${WHITE_LEVEL},
      "patch_size": 128,
      "patches_per_scene": 4,
      "tile_stride": 64,
      "context_pad": 0,
      "batch_size": 8,
      "num_workers": ${NUM_WORKERS},
      "pixel_weight": 2.0,
      "structure_weight": 1.0,
      "detail_weight": 0.5,
      "chroma_residual_weight": 1.0,
      "range_weight": 0.1,
      "confidence_floor": 0.25,
      "range_margin": 0.02,
      "charbonnier_epsilon": 0.001,
      "gradient_window": 7,
      "noise_floor": 0.0001,
      "saturation_start": 0.98,
      "saturation_threshold": 0.995,
      "innovation_weight": 2.0,
      "saturation_penalty": 6.0,
      "selection_temperature": 0.25,
      "reliability_guided_radius": 5,
      "reliability_guided_epsilon": 0.001,
      "strong_selection_threshold": 0.8,
      "morph_open_size": 3,
      "morph_close_size": 5,
      "restore_selection_threshold": 0.95,
      "chroma_black_valid_start": 0.001,
      "chroma_black_valid_end": 0.012,
      "chroma_guided_radius": 3,
      "chroma_guided_epsilon": 0.002,
      "chroma_agreement_gate": false,
      "chroma_agreement_tolerance": 0.5
    }
  }
}
JSON

echo "[4/4] Starting FinalFusion MEF training..."
cd "${AUTODL_ROOT}"
TRAIN_ARGS=(python -m finalfusion.train --config "${CONFIG_PATH}" --tasks mef)
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  echo "Resuming checkpoint: ${RUN_DIR}/last.pt"
  TRAIN_ARGS+=(--resume "${RUN_DIR}/last.pt")
fi
"${TRAIN_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train_console.log"
