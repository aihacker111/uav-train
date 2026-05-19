#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# train_hybrid.sh — Train HybridCenterNetDETR on VisDrone
#
# Usage:
#   bash scripts/train_hybrid.sh [options]
#
# Options:
#   --gpu        GPU index(es), comma-separated          default: 3
#   --arch       Model architecture                      default: hybrid_tiny
#   --bs         Batch size per GPU                      default: 8
#   --input_wh   Input resolution as W,H                default: 1280,704
#   --epochs     Total training epochs                   default: 100
#   --exp_id     Experiment name (auto-generated if not set)
#   --load_model Path to checkpoint to load
#   --resume     Resume from last checkpoint (flag)
#   --id_weight  0=detection only, 1=detection+ReID     default: 1
#
# Examples:
#   bash scripts/train_hybrid.sh
#   bash scripts/train_hybrid.sh --gpu 0,1 --arch hybrid_small
#   bash scripts/train_hybrid.sh --id_weight 0
#   bash scripts/train_hybrid.sh --resume
#   bash scripts/train_hybrid.sh --arch hybrid_tiny --bs 4 --input_wh 1088,640
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"
PRETRAIN_DIR="$ROOT_DIR/lwdetr_coco_pretrained"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPU="${GPU:-3}"
ARCH="${ARCH:-hybrid_tiny}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INPUT_WH="${INPUT_WH:-1280,704}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
EXP_ID="${EXP_ID:-hybrid_tiny_visdrone_reid}"
RESUME="${RESUME:-false}"
LOAD_MODEL="${LOAD_MODEL:-}"
ID_WEIGHT="${ID_WEIGHT:-1}"

# ── Parse named arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2";        shift 2 ;;
        --arch)       ARCH="$2";       shift 2 ;;
        --bs)         BATCH_SIZE="$2"; shift 2 ;;
        --input_wh)   INPUT_WH="$2";   shift 2 ;;
        --epochs)     NUM_EPOCHS="$2"; shift 2 ;;
        --exp_id)     EXP_ID="$2";     shift 2 ;;
        --load_model) LOAD_MODEL="$2"; shift 2 ;;
        --id_weight)  ID_WEIGHT="$2";  shift 2 ;;
        --resume)     RESUME="true";   shift 1 ;;
        *) echo "[WARN] Unknown argument: $1"; shift 1 ;;
    esac
done

# ── Derived values ─────────────────────────────────────────────────────────────
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | wc -l | tr -d ' ')

case "$ARCH" in
    hybrid_tiny)  PRETRAIN_WEIGHTS="$PRETRAIN_DIR/LWDETR_tiny_60e_coco.pth"  ;;
    hybrid_small) PRETRAIN_WEIGHTS="$PRETRAIN_DIR/LWDETR_small_60e_coco.pth" ;;
    hybrid_base)  PRETRAIN_WEIGHTS="$PRETRAIN_DIR/LWDETR_base_60e_coco.pth"  ;;
    *)            PRETRAIN_WEIGHTS="" ;;
esac

BACKBONE_ARG=""
if [[ "$RESUME" != "true" ]] && [[ -z "$LOAD_MODEL" ]] && [[ -f "$PRETRAIN_WEIGHTS" ]]; then
    BACKBONE_ARG="--backbone_weights ${PRETRAIN_WEIGHTS}"
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
echo "  Arch       : $ARCH"
echo "  GPU        : $GPU  (${NUM_GPUS} device(s))"
echo "  Batch/GPU  : $BATCH_SIZE   effective: $((BATCH_SIZE * NUM_GPUS * 4)) (×grad_accum 4)"
echo "  Input WxH  : $INPUT_WH"
echo "  Epochs     : $NUM_EPOCHS"
echo "  ReID       : id_weight=$ID_WEIGHT"
echo "  Pretrain   : ${PRETRAIN_WEIGHTS:-none}"
echo "  Output     : $SRC_DIR/exp/hybrid/$EXP_ID/"
echo "  Log        : $ROOT_DIR/logs/${EXP_ID}.log"
echo "════════════════════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$SRC_DIR"

python train.py \
    \
    --task              hybrid                          \
    --arch              "$ARCH"                         \
    --exp_id            "$EXP_ID"                       \
    $RESUME_ARG                                         \
    $LOAD_MODEL_ARG                                     \
    $BACKBONE_ARG                                       \
    \
    --data_cfg          ../src/lib/cfg/visdrone.json    \
    --num_classes       7                               \
    --input-wh          "$INPUT_WH"                     \
    \
    --gpus              "$GPU"                          \
    --num_workers       8                               \
    --batch_size        "$BATCH_SIZE"                   \
    --grad_accum        4                               \
    --use_amp                                           \
    --grad_checkpoint                                   \
    \
    --lr                4e-4                            \
    --base_batch_size   8                               \
    --lr_scale          none                            \
    --cosine_lr                                         \
    --warmup_iters      500                             \
    --min_lr_ratio      0.01                            \
    --backbone_lr_scale 0.1                             \
    --freeze_backbone_epochs 2                          \
    --grad_clip         1.0                             \
    \
    --num_epochs        "$NUM_EPOCHS"                   \
    --close_mosaic_epochs 10                            \
    --val_intervals     5                               \
    \
    --use_gumbel                                        \
    --tau_start         1.0                             \
    --tau_end           0.1                             \
    \
    --stage1_weight     2.0                             \
    --stage2_weight     1.0                             \
    --consist_weight    0.02                            \
    --consist_warmup_epochs 5                           \
    \
    --bbox_weight       5.0                             \
    --giou_weight       2.0                             \
    --hm_weight         1.0                             \
    --wh_weight         0.1                             \
    --off_weight        1.0                             \
    \
    --id_weight         "$ID_WEIGHT"                    \
    --reid_dim          256                             \
    --reid_cls_ids      0,1,2,3,4,5,6                  \
    \
    --K                 200                             \
    --num_output_levels 1                               \
    \
    --use_repeat_sampling                               \
    --repeat_thresh     0.001                           \
    \
    2>&1 | tee "$ROOT_DIR/logs/${EXP_ID}.log"
