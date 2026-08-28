#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/fusionkd/dual_teacher_r50_atss_glip_fpn_1x_coco_fewshot_jp.py"
GPUS=4

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

runs=(
    "1 2" "1 4" "1 5" "1 7" "1 8"
    "5 1" "5 2" "5 3" "5 5" "5 8"
    "10 3" "10 6" "10 7" "10 8" "10 9"
)

cd "${ROOT_DIR}"
for run in "${runs[@]}"; do
    read -r shot seed <<< "${run}"
    ann_file="fewshot/fewshot_from_train/shots${shot}/seed${seed}/train.json"
    work_dir="work_dirs/dual_teacher_jp6cls_${shot}shot_seed${seed}_800x800_60e"

    if [[ ! -f "datasets/1000_original_6class_0improve/y/${ann_file}" ]]; then
        echo "Missing annotation: datasets/1000_original_6class_0improve/y/${ann_file}" >&2
        exit 1
    fi

    echo "Starting shots=${shot}, seed=${seed}, work_dir=${work_dir}"
    if ! PORT="${PORT:-29500}" bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
        --work-dir "${work_dir}" \
        --cfg-options \
        "train_dataloader.dataset.datasets.0.ann_file=${ann_file}"; then
        echo "Training failed for shots=${shot}, seed=${seed}; stopping remaining runs." >&2
        exit 1
    fi
done
