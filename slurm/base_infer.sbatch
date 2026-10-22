#!/bin/bash
#SBATCH -J mindeye-base
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh
mkdir -p $OUT_ROOT
source fmri/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd src
# Any GPU type with >= ~46 GB (peak mem measured 32.5 GB on an H200); an L40S is ~2-3x slower than an H200, hence the 4 h limit.
# Overrides: sbatch -t 01:00:00 --export=ALL,OUT_NAME=base_smoke,MAX_IMAGES=10 slurm/base_infer.sbatch   (RESUME=1 resumes finished shards)
python -u interp/recon_infer.py --data_path $NSD_DATA --cache_dir $NSD_DATA --ckpt_root $CKPT_DIR \
  --unclip_ckpt $CKPT_DIR/unclip6_epoch0_step110000.ckpt --bigg_safetensors $BIGG_SAFETENSORS \
  --out_dir $OUT_ROOT --out_name ${OUT_NAME:-base} --no-blurry_recon --max_images ${MAX_IMAGES:-0} \
  ${RESUME:+--resume}
