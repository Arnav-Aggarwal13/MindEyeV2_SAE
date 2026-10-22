#!/bin/bash
#SBATCH -J mindeye-cliptarget
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
cd src
python -u interp/make_clip_target.py --data_path $NSD_DATA --bigg_safetensors $BIGG_SAFETENSORS \
  --out_dir $OUT_ROOT --out_name ${OUT_NAME:-base}
