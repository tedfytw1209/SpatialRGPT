#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=${3:-"vicuna_v1"}
QUESTION_FILE=${4:-"./playground/data/eval/iu_xray/simple_questions.jsonl"}
IMAGE_DIR=${5:-"/orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized"}
DATASET_NAME=${6:-"iu_xray"}
BS=${7:-"16"}
METADATA=${8:-"None"}

echo "$MODEL_PATH $CKPT"

source activate base
conda activate vila

# This script runs a batch evaluation for a Visual Question Answering (VQA) model using the llava.eval.model_vqa_batch module.
# 
# Environment Variables:
#   CUDA_VISIBLE_DEVICES: Specifies which GPU to use.
#
# Arguments:
#   --model-path: Path to the pre-trained model.
#   --question-file: Path to the file containing the questions.
#   --meta-path: Path to the metadata file.
#   --replace-q: Flag to indicate whether to replace questions.
#   --image-folder: Directory containing the images.
#   --answers-file: Path to the output file where the answers will be saved.
#   --batch-size: Number of samples per batch.
#   --temperature: Sampling temperature for the model.
#   --conv-mode: Conversation mode for the model.
python -m llava.eval.model_vqa_batch \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --meta-path $METADATA \
    --replace-q \
    --image-folder $IMAGE_DIR \
    --answers-file ./eval_output/$DATASET_NAME/answers_$CKPT.jsonl \
    --batch-size $BS \
    --temperature 0.2 \
    --conv-mode $CONV_MODE
