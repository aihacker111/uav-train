#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# train_deim_mot.sh — Train DEIMMotNet (backbone + encoder + CenterNet/ReID)
#
# Usage:
#   bash scripts/train_deim_mot.sh [options]
#
# Options:
#   --gpu         GPU index(es), comma-separated          default: 0
#   --deim_config Path to DEIM-UAV YAML config            default: deimv2_hgnetv2_s_coco.yml
#   --bs          Batch size per GPU                      default: 4
#   --input_wh    Input resolution as W,H                 default: 1280,704
#   --epochs      Total training epochs                   default: 30
#   --exp_id      Experiment name
#   --load_model  Path to checkpoint to resume from
#   --deim_ckpt   Path to DEIMv2 COCO pretrained weights (.pth)
#   --resume      Resume from last checkpoint (flag)
#   --id_weight   0=detection only, 1=detection+ReID      default: 1
#   --reid_dim    ReID embedding dimension                 default: 128
#
# Examples:
#   bash scripts/train_deim_mot.sh
#   bash scripts/train_deim_mot.sh --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_n_coco.yml
#   bash scripts/train_deim_mot.sh --gpu 0,1 --bs 8
#   bash scripts/train_deim_mot.sh --deim_ckpt ../deimv2_coco_pretrained/hgnetv2_s.pth
#   bash scripts/train_deim_mot.sh --resume
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPU="${GPU:-0}"
DEIM_CONFIG="${DEIM_CONFIG:-lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INPUT_WH="${INPUT_WH:-1280,704}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
EXP_ID="${EXP_ID:-deim_mot_hgnetv2_s_visdrone}"
RESUME="${RESUME:-false}"
LOAD_MODEL="${LOAD_MODEL:-}"
DEIM_CKPT="${DEIM_CKPT:-}"
ID_WEIGHT="${ID_WEIGHT:-1}"
REID_DIM="${REID_DIM:-128}"

# ── Parse named arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2";         shift 2 ;;
        --deim_config) DEIM_CONFIG="$2"; shift 2 ;;
        --bs)          BATCH_SIZE="$2";  shift 2 ;;
        --input_wh)    INPUT_WH="$2";    shift 2 ;;
        --epochs)      NUM_EPOCHS="$2";  shift 2 ;;
        --exp_id)      EXP_ID="$2";      shift 2 ;;
        --load_model)  LOAD_MODEL="$2";  shift 2 ;;
        --deim_ckpt)   DEIM_CKPT="$2";  shift 2 ;;
        --id_weight)   ID_WEIGHT="$2";   shift 2 ;;
        --reid_dim)    REID_DIM="$2";    shift 2 ;;
        --resume)      RESUME="true";    shift 1 ;;
        *) echo "[WARN] Unknown argument: $1"; shift 1 ;;
    esac
done

# ── Derived values ─────────────────────────────────────────────────────────────
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | wc -l | tr -d ' ')

BACKBONE_ARG=""
if [[ "$RESUME" != "true" ]] && [[ -z "$LOAD_MODEL" ]] && [[ -n "$DEIM_CKPT" ]] && [[ -f "$DEIM_CKPT" ]]; then
    BACKBONE_ARG="--backbone_weights ${DEIM_CKPT}"
fi

RESUME_ARG=""
[[ "$RESUME" == "true" ]] && RESUME_ARG="--resume"

LOAD_MODEL_ARG=""
[[ -n "$LOAD_MODEL" ]] && LOAD_MODEL_ARG="--load_model ${LOAD_MODEL}"

# ── Log dir ───────────────────────────────────────────────────────────────────
mkdir -p "$ROOT_DIR/logs"

# ── Print config summary ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Experiment : $EXP_ID"
echo "  Arch       : deim_mot"
echo "  DEIM cfg   : $DEIM_CONFIG"
echo "  GPU        : $GPU  (${NUM_GPUS} device(s))"
echo "  Batch/GPU  : $BATCH_SIZE"
echo "  Input WxH  : $INPUT_WH"
echo "  Epochs     : $NUM_EPOCHS"
echo "  ReID       : id_weight=$ID_WEIGHT  reid_dim=$REID_DIM"
echo "  DEIM ckpt  : ${DEIM_CKPT:-none}"
echo "  Output     : $SRC_DIR/exp/deim_mot/$EXP_ID/"
echo "  Log        : $ROOT_DIR/logs/${EXP_ID}.log"
echo "════════════════════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$SRC_DIR"

python train.py \
    \
    --task              deim_mot                        \
    --arch              deim_mot                        \
    --deim_config       "$DEIM_CONFIG"                  \
    --exp_id            "$EXP_ID"                       \
    $RESUME_ARG                                         \
    $LOAD_MODEL_ARG                                     \
    $BACKBONE_ARG                                       \
    \
    --data_cfg          ../src/lib/cfg/visdrone.json    \
    --input-wh          "$INPUT_WH"                     \
    \
    --gpus              "$GPU"                          \
    --num_workers       8                               \
    --batch_size        "$BATCH_SIZE"                   \
    --use_amp                                           \
    \
    --lr                4e-4                            \
    --base_batch_size   4                               \
    --lr_scale          linear                          \
    --cosine_lr                                         \
    --warmup_iters      500                             \
    --min_lr_ratio      0.01                            \
    --backbone_lr_scale 0.1                             \
    --freeze_backbone_epochs 2                          \
    --grad_clip         0.5                             \
    \
    --num_epochs        "$NUM_EPOCHS"                   \
    --close_mosaic_epochs 5                             \
    --val_intervals     5                               \
    \
    --hm_weight         1.0                             \
    --wh_weight         0.1                             \
    --off_weight        1.0                             \
    --iou_weight        1.0                             \
    \
    --id_weight         "$ID_WEIGHT"                    \
    --reid_dim          "$REID_DIM"                     \
    --id_loss           ce                              \
    \
    --K                 200                             \
    \
    2>&1 | tee "$ROOT_DIR/logs/${EXP_ID}.log"
