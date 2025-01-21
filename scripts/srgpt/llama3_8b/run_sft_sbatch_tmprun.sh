#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=bianjiang
#SBATCH --qos=bianjiang

date;hostname;pwd

module load singularity

# Run a tutorial python script within the container. Modify the path to your container and your script.

#RGPT PART
#cd /orange/bianjiang/tienyu/SpatialRGPT/

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/llama3_8b/3_sft_medical.sh \
 /orange/bianjiang/tienyu/vila_models/srgpt_0ep_m3vision/ \
 ldlcct2f_regm3v_lr2_2ep ldct2_vqa_f_reg+lcct2_vqa_f_reg 2 /orange/bianjiang/tienyu/vila_models/ 2e-5

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/llama3_8b/3_sft_medical.sh \
 /orange/bianjiang/tienyu/vila_models/srgpt_0ep_m3vision/ \
 ldct2f_regm3v_lr2_2ep ldct2_vqa_f_reg 2 /orange/bianjiang/tienyu/vila_models/ 2e-5

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/llama3_8b/3_sft_medical.sh \
 /orange/bianjiang/tienyu/vila_models/srgpt_0ep_m3vision/ \
 ldlcct2f_regm3v_wunk_lr2_2ep ldct2_vqa_wunk_reg+lcct2_vqa_wunk_reg 2 /orange/bianjiang/tienyu/vila_models/ 2e-5

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/llama3_8b/3_sft_medical.sh \
 /orange/bianjiang/tienyu/vila_models/srgpt_0ep_m3vision/ \
 ldct2f_regm3v_wunk_lr2_2ep ldct2_vqa_wunk_reg 2 /orange/bianjiang/tienyu/vila_models/ 2e-5
