#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

# OUTPUT of stage 2 script
n_node=1

# OUTPUT of stage 2 script
STAGE2_PATH=$1
# Final output checkpoint path
OUTPUT=$2
# data mixture
DATA=${3:-"iu_xray"}
# epoch
EPOCH=${4:-"5"}
#model save place
OUTPUT_DIR=${5:-"./checkpoints"}
#learning rate
LR=${6:-"1e-4"}
#bs
bs=${7:-"16"}

echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

source activate base
conda activate srgpt

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $STAGE2_PATH \
    --version llama_3 \
    --data_mixture $DATA \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --enable_region True \
    --enable_depth False \
    --region_extractor regiongpt \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_region_extractor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR/$OUTPUT \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
