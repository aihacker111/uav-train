#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# train_hybrid.sh — Train HybridDETR on VisDrone
#
# Usage:
#   bash scripts/train_hybrid.sh [options]
#
# Core options:
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
# DN training options:
#   --num_dn_groups         Noised GT copies per image  default: 5
#   --dn_label_noise_ratio  Label noise fraction        default: 0.5
#   --dn_box_noise_scale    Box noise magnitude         default: 0.4
#   --dn_max_queries        Per-image DN query cap      default: 500
#   --dn_l1_weight          DN L1 loss weight           default: 1.0
#   --dn_cls_weight         DN class loss weight        default: 1.0
#   --no_dn                 Disable DN training (flag)
#
# Query generation options:
#   --K                     Top-K detect queries        default: 200
#   --use_spatial_partition Enable SP query gen (flag)  default: off
#   --sp_grid_rows          SP grid rows                default: 4
#   --sp_grid_cols          SP grid cols                default: 4
#   --sp_queries_per_region Local queries per region    default: 50
#   --sp_global_queries     Global queries              default: 32
#
# Scorer options:
#   --scorer_head_conv      Scorer conv channels        default: 64
#   --no_multiscale_fusion  Disable s16→s8 fusion (flag)
#
# Examples:
#   bash scripts/train_hybrid.sh
#   bash scripts/train_hybrid.sh --gpu 0,1 --arch hybrid_small
#   bash scripts/train_hybrid.sh --id_weight 0
#   bash scripts/train_hybrid.sh --resume
#   bash scripts/train_hybrid.sh --arch hybrid_tiny --bs 4 --input_wh 1088,640
#   bash scripts/train_hybrid.sh --no_dn                              # disable DN
#   bash scripts/train_hybrid.sh --num_dn_groups 3 --dn_l1_weight 2.0
#   bash scripts/train_hybrid.sh --use_spatial_partition --K 200
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
K="${K:-200}"

# DN defaults
NUM_DN_GROUPS="${NUM_DN_GROUPS:-5}"
DN_LABEL_NOISE="${DN_LABEL_NOISE:-0.5}"
DN_BOX_NOISE="${DN_BOX_NOISE:-0.4}"
DN_MAX_QUERIES="${DN_MAX_QUERIES:-500}"
DN_L1_WEIGHT="${DN_L1_WEIGHT:-1.0}"
DN_CLS_WEIGHT="${DN_CLS_WEIGHT:-1.0}"
NO_DN="${NO_DN:-false}"

# Spatial partition defaults
USE_SP="${USE_SP:-false}"
SP_GRID_ROWS="${SP_GRID_ROWS:-4}"
SP_GRID_COLS="${SP_GRID_COLS:-4}"
SP_QUERIES_PER_REGION="${SP_QUERIES_PER_REGION:-50}"
SP_GLOBAL_QUERIES="${SP_GLOBAL_QUERIES:-32}"

# Scorer defaults
SCORER_HEAD_CONV="${SCORER_HEAD_CONV:-64}"
NO_MULTISCALE_FUSION="${NO_MULTISCALE_FUSION:-false}"

# ── Parse named arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)                    GPU="$2";                    shift 2 ;;
        --arch)                   ARCH="$2";                   shift 2 ;;
        --bs)                     BATCH_SIZE="$2";             shift 2 ;;
        --input_wh)               INPUT_WH="$2";               shift 2 ;;
        --epochs)                 NUM_EPOCHS="$2";             shift 2 ;;
        --exp_id)                 EXP_ID="$2";                 shift 2 ;;
        --load_model)             LOAD_MODEL="$2";             shift 2 ;;
        --id_weight)              ID_WEIGHT="$2";              shift 2 ;;
        --K)                      K="$2";                      shift 2 ;;
        --resume)                 RESUME="true";               shift 1 ;;
        # DN
        --num_dn_groups)          NUM_DN_GROUPS="$2";          shift 2 ;;
        --dn_label_noise_ratio)   DN_LABEL_NOISE="$2";         shift 2 ;;
        --dn_box_noise_scale)     DN_BOX_NOISE="$2";           shift 2 ;;
        --dn_max_queries)         DN_MAX_QUERIES="$2";         shift 2 ;;
        --dn_l1_weight)           DN_L1_WEIGHT="$2";           shift 2 ;;
        --dn_cls_weight)          DN_CLS_WEIGHT="$2";          shift 2 ;;
        --no_dn)                  NO_DN="true";                shift 1 ;;
        # Spatial partition
        --use_spatial_partition)  USE_SP="true";               shift 1 ;;
        --sp_grid_rows)           SP_GRID_ROWS="$2";           shift 2 ;;
        --sp_grid_cols)           SP_GRID_COLS="$2";           shift 2 ;;
        --sp_queries_per_region)  SP_QUERIES_PER_REGION="$2";  shift 2 ;;
        --sp_global_queries)      SP_GLOBAL_QUERIES="$2";      shift 2 ;;
        # Scorer
        --scorer_head_conv)       SCORER_HEAD_CONV="$2";       shift 2 ;;
        --no_multiscale_fusion)   NO_MULTISCALE_FUSION="true"; shift 1 ;;
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

# DN: disable by setting num_dn_groups=0
if [[ "$NO_DN" == "true" ]]; then
    NUM_DN_GROUPS=0
fi

SP_ARG=""
[[ "$USE_SP" == "true" ]] && SP_ARG="--use_spatial_partition"

MULTISCALE_ARG="--use_multiscale_fusion"
[[ "$NO_MULTISCALE_FUSION" == "true" ]] && MULTISCALE_ARG="--no-use_multiscale_fusion"

# SP query count for display
if [[ "$USE_SP" == "true" ]]; then
    TOTAL_SP_Q=$(( SP_GRID_ROWS * SP_GRID_COLS * SP_QUERIES_PER_REGION + SP_GLOBAL_QUERIES ))
    Q_DISPLAY="${TOTAL_SP_Q} (SP: ${SP_GRID_ROWS}×${SP_GRID_COLS}×${SP_QUERIES_PER_REGION}+${SP_GLOBAL_QUERIES})"
else
    Q_DISPLAY="${K}"
fi

DN_DISPLAY="groups=${NUM_DN_GROUPS}  noise_lbl=${DN_LABEL_NOISE}  noise_box=${DN_BOX_NOISE}  max_q=${DN_MAX_QUERIES}"
[[ "$NO_DN" == "true" ]] && DN_DISPLAY="disabled"

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
echo "  Queries    : $Q_DISPLAY"
echo "  DN         : $DN_DISPLAY"
echo "  DN weights : l1=${DN_L1_WEIGHT}  cls=${DN_CLS_WEIGHT}"
echo "  Scorer     : head_conv=${SCORER_HEAD_CONV}  multiscale_fusion=$([ "$NO_MULTISCALE_FUSION" == "true" ] && echo off || echo on)"
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
    --K                 "$K"                            \
    --num_output_levels 1                               \
    $SP_ARG                                             \
    --sp_grid_rows      "$SP_GRID_ROWS"                 \
    --sp_grid_cols      "$SP_GRID_COLS"                 \
    --sp_queries_per_region "$SP_QUERIES_PER_REGION"    \
    --sp_global_queries "$SP_GLOBAL_QUERIES"            \
    \
    --scorer_head_conv  "$SCORER_HEAD_CONV"             \
    $MULTISCALE_ARG                                     \
    \
    --num_dn_groups         "$NUM_DN_GROUPS"            \
    --dn_label_noise_ratio  "$DN_LABEL_NOISE"           \
    --dn_box_noise_scale    "$DN_BOX_NOISE"             \
    --dn_max_queries        "$DN_MAX_QUERIES"           \
    --dn_l1_weight          "$DN_L1_WEIGHT"             \
    --dn_cls_weight         "$DN_CLS_WEIGHT"            \
    \
    --use_repeat_sampling                               \
    --repeat_thresh     0.001                           \
    \
    2>&1 | tee "$ROOT_DIR/logs/${EXP_ID}.log"
