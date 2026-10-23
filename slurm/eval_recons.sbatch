#!/bin/bash
#SBATCH -J mindeye-eval
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3006   # CUDA init fails on this node (seen twice)
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
unset HF_HUB_OFFLINE   # torch.hub (SwAV) may need github
cd src
# Overrides: sbatch --export=ALL,RECONS_PATH=...,EVAL_NAME=... slurm/eval_recons.sbatch
python -u interp/eval_recons.py --data_path $NSD_DATA \
  --recons_path ${RECONS_PATH:-$NSD_DATA/evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_recons.pt} \
  --betas_avg $OUT_ROOT/base/base_test_betas_avg.pt \
  --out_dir $OUT_ROOT --out_name ${EVAL_NAME:-eval_authors_recons} ${EXTRA_ARGS:-}
