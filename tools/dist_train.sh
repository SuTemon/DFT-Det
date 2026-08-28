# #!/usr/bin/env bash
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# CUDA_VISIBLE_DEVICES=1


# CONFIG="configs/fusionkd/fusionkd_r50_atss_glip_fpn_1x_coco.py"  #fusion-kd-atss

# GPUS=4
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:3} \



# # --resume "work_dirs/jojo_glip_res/epoch_11.pth" \
# #--resume "work_dirs/jojo_glip_res/epoch_24.pth" \

# export PYTHONPATH=$PYTHONPATH:$(pwd)

#!/usr/bin/env bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 接收 Python 传进来的前两个参数
CONFIG=$1
GPUS=$2

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

export PYTHONPATH=$PYTHONPATH:$(pwd)