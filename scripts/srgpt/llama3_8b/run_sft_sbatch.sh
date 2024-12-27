#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=bianjiang
#SBATCH --qos=bianjiang

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

date;hostname;pwd

module load singularity

# Run a tutorial python script within the container. Modify the path to your container and your script.
singularity exec --nv 3_sft_medical.sh $STAGE2_PATH $OUTPUT $DATA $EPOCH $OUTPUT_DIR $LR $bs
