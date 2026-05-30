#!/usr/bin/env bash
# Run DEIMv2-DINOv3-S tracking inference on VisDrone test-dev sequences.
# Model is loaded via DEIMv2 YAMLConfig — no separate create_model wrapper.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/src"
CONFIG="${SRC}/configs/deimv2_dinov3_s_visdrone.yml"
MODEL="${REPO_ROOT}/outputs/deimv2_dinov3_s_visdrone/best_stg2.pth"
DATA_DIR="/media/jianbo/ioe/UAVdata"    # update to your data root

cd "${SRC}"

python track_AMOT.py \
    --deim_config "${CONFIG}" \
    --load_model  "${MODEL}" \
    --data_dir    "${DATA_DIR}" \
    --gpus        0 \
    --input-wh    "1280,704" \
    --num_classes 10 \
    --K           500 \
    --conf_thres  0.4 \
    --nms_thres   0.4 \
    --track_buffer 30 \
    --min-box-area 100 \
    --test_visdrone True \
    --save_dir_result deimv2_dinov3_s_visdrone \
    "$@"
