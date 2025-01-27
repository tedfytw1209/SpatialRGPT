#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=bianjiang
#SBATCH --qos=bianjiang

date;hostname;pwd

module load singularity

# Run a tutorial python script within the container. Modify the path to your container and your script.
singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldlcct2f_regm3v_lr2_2ep/ ldlcct2f_regm3v_lr2_2ep_ldtest llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_bsize.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldct2f_regm3v_lr2_2ep/ ldct2f_regm3v_lr2_2ep_ldtest llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_bsize.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldlcct2f_regm3v_wunk_lr2_2ep/ ldlcct2f_regm3v_wunk_lr2_2ep_ldtest llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_bsize.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldct2f_regm3v_wunk_lr2_2ep/ ldct2f_regm3v_wunk_lr2_2ep_ldtest llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_bsize.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldlcct2f_regm3v_lr2_2ep/ ldlcct2f_regm3v_lr2_2ep_ldnorm llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_norm.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldct2f_regm3v_lr2_2ep/ ldct2f_regm3v_lr2_2ep_ldnorm llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_norm.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldlcct2f_regm3v_wunk_lr2_2ep/ ldlcct2f_regm3v_wunk_lr2_2ep_ldnorm llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_norm.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct

singularity exec --nv /blue/bianjiang/tienyuchang/anaconda bash scripts/srgpt/eval/eval_vqa_run.sh /orange/bianjiang/tienyu/vila_models/ldct2f_regm3v_wunk_lr2_2ep/ ldct2f_regm3v_wunk_lr2_2ep_ldnorm llama_3 ./data/LungCancer_CTv2/ldct2f_region_test_norm.json /orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/ lc_ct
