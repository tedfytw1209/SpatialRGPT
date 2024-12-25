#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

DATA_FOLDER=/orange/bianjiang/tienyu/coco2017/

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_region_cls \
        --model-path $MODEL_PATH \
        --annotation-file /orange/bianjiang/tienyu/coco2017/annotations/instances_val2017.json \
        --image-folder $DATA_FOLDER \
        --answers-file ./eval_output/$CKPT/regiongpt/coco/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --dataset coco &
done

wait

output_file=./eval_output/$CKPT/regiongpt/coco/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval_output/$CKPT/regiongpt/coco/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/srgpt/eval/eval_coco_obo.py $output_file
