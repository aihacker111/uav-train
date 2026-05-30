#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train DEIMv2-DINOv3-S on VisDrone  —  100 % DEIMv2 pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# Pre-requisites (download once):
#   DEIMv2-S COCO ckpt : https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL
#   → place at  models/deimv2_dinov3_s_coco.pth
#
# Usage
# ──────
#  Fine-tune from COCO pretrained (recommended first run):
#    bash scripts/train_deimv2_visdrone.sh
#
#  Resume an interrupted run:
#    bash scripts/train_deimv2_visdrone.sh --resume
#
#  Override epochs / batch size from CLI:
#    bash scripts/train_deimv2_visdrone.sh -u epoches=60 train_dataloader.total_batch_size=8
#
#  Multi-GPU (4 GPUs):
#    N_GPU=4 bash scripts/train_deimv2_visdrone.sh
#
#  Eval-only:
#    bash scripts/train_deimv2_visdrone.sh --test-only -r outputs/deimv2_dinov3_s_visdrone/best_stg2.pth
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/src"
CONFIG="${SRC}/configs/deimv2_dinov3_s_visdrone.yml"
PRETRAINED="${REPO_ROOT}/models/deimv2_dinov3_s_coco.pth"
OUTPUT_DIR="${REPO_ROOT}/outputs/deimv2_dinov3_s_visdrone"

# ── GPU setup ─────────────────────────────────────────────────────────────────
N_GPU="${N_GPU:-1}"   # override: N_GPU=4 bash train_deimv2_visdrone.sh

# ── Parse --resume / --test-only flags from $@ so we can switch -t ↔ -r ─────
USE_RESUME=0
TEST_ONLY=0
RESUME_CKPT=""

PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            USE_RESUME=1
            # default resume path — override with -r <path> below
            RESUME_CKPT="${OUTPUT_DIR}/last.pth"
            shift ;;
        -r|--resume=*)
            USE_RESUME=1
            if [[ "$1" == "-r" ]]; then
                RESUME_CKPT="$2"; shift 2
            else
                RESUME_CKPT="${1#*=}"; shift
            fi ;;
        --test-only)
            TEST_ONLY=1
            PASS_ARGS+=("--test-only")
            shift ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ $USE_RESUME -eq 0 && ! -f "${PRETRAINED}" ]]; then
    echo "ERROR: pretrained checkpoint not found: ${PRETRAINED}"
    echo "  Download from https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL"
    echo "  and place at  models/deimv2_dinov3_s_coco.pth"
    exit 1
fi

if [[ $USE_RESUME -eq 1 && ! -f "${RESUME_CKPT}" ]]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CKPT}"
    exit 1
fi

# ── Source / checkpoint flag ──────────────────────────────────────────────────
if [[ $USE_RESUME -eq 1 ]]; then
    CKPT_FLAG=(-r "${RESUME_CKPT}")
    echo "==> Resuming from  ${RESUME_CKPT}"
else
    CKPT_FLAG=(-t "${PRETRAINED}")
    echo "==> Fine-tuning from  ${PRETRAINED}"
fi

echo "    config     : ${CONFIG}"
echo "    output_dir : ${OUTPUT_DIR}"
echo "    n_gpu      : ${N_GPU}"
[[ ${#PASS_ARGS[@]} -gt 0 ]] && echo "    extra args : ${PASS_ARGS[*]}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "${N_GPU}" -gt 1 ]]; then
    # Distributed training via torchrun
    torchrun \
        --nproc_per_node="${N_GPU}" \
        --master_port="${MASTER_PORT:-29500}" \
        "${SRC}/train_mot.py" \
            -c "${CONFIG}" \
            "${CKPT_FLAG[@]}" \
            --output-dir "${OUTPUT_DIR}" \
            "${PASS_ARGS[@]}"
else
    # Single GPU
    python "${SRC}/train_mot.py" \
        -c "${CONFIG}" \
        "${CKPT_FLAG[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        "${PASS_ARGS[@]}"
fi

echo ""
echo "==> Done. Checkpoints in: ${OUTPUT_DIR}"
