#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_track.sh — Run VisDrone tracking evaluation with track_AMOT.py
#
# Usage:
#   bash scripts/run_track.sh [MODEL_PATH] [ARCH] [GPU]
#
# Examples:
#   bash scripts/run_track.sh                                          # defaults
#   bash scripts/run_track.sh exp/hybrid/my_exp/model_last.pth
#   bash scripts/run_track.sh exp/hybrid/my_exp/model_last.pth hybrid_small 1
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurable defaults (override via positional args or env vars) ──────────
MODEL=${1:-"exp/hybrid/visdrone_hybrid_small_2gpu/model_last.pth"}
ARCH=${2:-"hybrid_small"}
GPU=${3:-"0"}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"

# Resolve model path: treat as relative to project root if not absolute
if [[ "$MODEL" != /* ]]; then
    MODEL="$ROOT_DIR/$MODEL"
fi

# Root directory that contains VisDrone2019/test_dev/sequences/
DATA_DIR="${DATA_DIR:-/media/jianbo/ioe/UAVdata}"

# ── Derived from arch ─────────────────────────────────────────────────────────
if [[ "$ARCH" == hybrid* ]]; then
    TASK="hybrid"
else
    TASK="mot"
fi

# ── Tracking thresholds ───────────────────────────────────────────────────────
CONF_THRES="${CONF_THRES:-0.4}"        # heatmap/score threshold for tracking
DET_THRES="${DET_THRES:-0.3}"          # detection confidence threshold
NMS_THRES="${NMS_THRES:-0.4}"          # NMS IoU threshold
TRACK_BUFFER="${TRACK_BUFFER:-30}"     # frames a lost track is kept alive
MIN_BOX_AREA="${MIN_BOX_AREA:-100}"    # filter out boxes smaller than this (px²)
K="${K:-200}"                          # max detections per frame

# Input resolution must match what the model was trained with
INPUT_WH="${INPUT_WH:-1280,704}"       # W,H — change to 1088,640 for tiny/fast

# ReID class IDs (0-indexed, comma-separated; must match training config)
REID_CLS_IDS="${REID_CLS_IDS:-0,1,2,3,4,5,6,7,8,9}"

# Save directory for result .txt files
SAVE_DIR_RESULT="${SAVE_DIR_RESULT:-track_results_$(date +%Y%m%d_%H%M%S)}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "        Set MODEL env var or pass it as the first argument."
    exit 1
fi

echo "========================================================"
echo "  Model      : $MODEL"
echo "  Arch       : $ARCH  (task=$TASK)"
echo "  GPU        : $GPU"
echo "  Data dir   : $DATA_DIR"
echo "  Input WxH  : $INPUT_WH"
echo "  conf/det   : $CONF_THRES / $DET_THRES"
echo "  Results    : $SAVE_DIR_RESULT"
echo "========================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$SRC_DIR"

python track_AMOT.py \
    --task          "$TASK"             \
    --arch          "$ARCH"             \
    --load_model    "$MODEL"            \
    --gpus          "$GPU"              \
    --data_dir      "$DATA_DIR"         \
    --input-wh      "$INPUT_WH"         \
    --reid_cls_ids  "$REID_CLS_IDS"     \
    --conf_thres    "$CONF_THRES"       \
    --det_thres     "$DET_THRES"        \
    --nms_thres     "$NMS_THRES"        \
    --track_buffer  "$TRACK_BUFFER"     \
    --min-box-area  "$MIN_BOX_AREA"     \
    --K             "$K"                \
    --save_dir_result "$SAVE_DIR_RESULT" \
    --test_visdrone True
