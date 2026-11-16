#!/usr/bin/env bash
set -Eeuo pipefail

# One command: FinalFusion MEF test inference followed by the enhanced fixed ISP.
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
FINALFUSION_DIR="${FINALFUSION_DIR:-${AUTODL_ROOT}/finalfusion}"
DATA_ROOT="${DATA_ROOT:-${AUTODL_ROOT}/DatasetMEF_denoised}"
METADATA_ROOT="${METADATA_ROOT:-${AUTODL_ROOT}/DatasetMEF_metadata_numeric}"
RUN_DIR="${RUN_DIR:-${AUTODL_ROOT}/runs/finalfusion_mef}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best.pt}"
CONFIG="${CONFIG:-${RUN_DIR}/mef_train_config.json}"
SPLIT_JSON="${SPLIT_JSON:-${FINALFUSION_DIR}/mef_scene_split.json}"
SPLIT="${SPLIT:-test}"
RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-${AUTODL_ROOT}/outputs/finalfusion_mef_raw}"
ISP_DIR="${ISP_DIR:-${AUTODL_ROOT}/opencv_fixed_raw_compare_isp}"
RGB_OUTPUT_ROOT="${RGB_OUTPUT_ROOT:-${AUTODL_ROOT}/outputs/finalfusion_mef_isp}"
DEVICE="${DEVICE:-cuda}"
TILE_SIZE="${TILE_SIZE:-512}"
HALO="${HALO:-32}"
FIXED_EV="${FIXED_EV:-0.0}"
FORCE="${FORCE:-0}"

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
require_dir "${METADATA_ROOT}"
require_dir "${ISP_DIR}"
require_file "${CHECKPOINT}"
require_file "${CONFIG}"
require_file "${SPLIT_JSON}"
require_file "${FINALFUSION_DIR}/infer_mef_batch.py"
require_file "${ISP_DIR}/render_finalfusion_mef.py"

mkdir -p "${RAW_OUTPUT_ROOT}" "${RGB_OUTPUT_ROOT}"
export PYTHONPATH="${AUTODL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

INFER_ARGS=(
  python -m finalfusion.infer_mef_batch
  --checkpoint "${CHECKPOINT}"
  --config "${CONFIG}"
  --split-json "${SPLIT_JSON}"
  --split "${SPLIT}"
  --output-root "${RAW_OUTPUT_ROOT}"
  --metadata-root "${METADATA_ROOT}"
  --extra-metadata-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --tile-size "${TILE_SIZE}"
  --halo "${HALO}"
)
ISP_ARGS=(
  python "${ISP_DIR}/render_finalfusion_mef.py"
  --fused-root "${RAW_OUTPUT_ROOT}"
  --output-root "${RGB_OUTPUT_ROOT}"
  --fixed-ev "${FIXED_EV}"
  --false-color-kernel 5
  --false-color-strength 0.85
  --highlight-start 0.90
  --highlight-end 0.995
)
# The first ISP release named the metadata-only argument --data-root.
# Accept either release so inference outputs never need to be regenerated.
if python "${ISP_DIR}/render_finalfusion_mef.py" --help 2>&1 | grep -q -- "--metadata-root"; then
  ISP_ARGS+=(--metadata-root "${METADATA_ROOT}")
else
  ISP_ARGS+=(--data-root "${METADATA_ROOT}")
fi
if [[ "${FORCE}" == "1" ]]; then
  INFER_ARGS+=(--force)
  ISP_ARGS+=(--force)
fi

echo "[1/2] Running FinalFusion MEF ${SPLIT} inference..."
cd "${AUTODL_ROOT}"
"${INFER_ARGS[@]}"

echo "[2/2] Rendering with per-scene DNG metadata, highlight recovery, ratio-preserving shoulder, and false-color suppression..."
"${ISP_ARGS[@]}"

echo "RAW outputs: ${RAW_OUTPUT_ROOT}"
echo "ISP outputs: ${RGB_OUTPUT_ROOT}"
