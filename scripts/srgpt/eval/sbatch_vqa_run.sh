#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=bianjiang
#SBATCH --qos=bianjiang

MODEL_PATH=$1
CKPT=$2
CONV_MODE=${3:-"vicuna_v1"}
QUESTION_FILE=${4:-"./playground/data/eval/iu_xray/simple_questions.json"}
IMAGE_DIR=${5:-"/orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized"}
DATASET_NAME=${6:-"iu_xray"}

date;hostname;pwd

module load singularity

# Run a tutorial python script within the container. Modify the path to your container and your script.
singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh $MODEL_PATH $CKPT $CONV_MODE $QUESTION_FILE $IMAGE_DIR $DATASET_NAME
