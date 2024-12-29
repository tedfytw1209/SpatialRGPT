#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=${3:-"vicuna_v1"}
QUESTION_FILE=${4:-"./playground/data/eval/iu_xray/simple_questions.json"}
IMAGE_DIR=${5:-"/orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized"}
DATASET_NAME=${6:-"iu_xray"}


echo "$MODEL_PATH $CKPT"

source activate base
conda activate vila

# This script runs a batch evaluation for a Visual Question Answering (VQA) model using the llava.eval.model_vqa_batch module.
# 
# Environment Variables:
#   CUDA_VISIBLE_DEVICES: Specifies which GPU to use.
python -m llava.eval.eval_region_vqa \
    --model-path $MODEL_PATH \
    --annotation-file $QUESTION_FILE \
    --image-folder $IMAGE_DIR \
    --answers-file ./eval_output/finetune/$DATASET_NAME/answers_$CKPT.json \
    --temperature 0.2 \
    --conv-mode $CONV_MODE
