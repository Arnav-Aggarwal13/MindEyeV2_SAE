#!/bin/bash
#SBATCH -J mindeye-diag
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
cd src
python -u interp/diag_clip_range.py --data_path $NSD_DATA --bigg_safetensors $BIGG_SAFETENSORS \
  --prior_out base_smoke_L40S=$OUT_ROOT/base_smoke/base_smoke_all_prior_out.pt smoke_base_H200=$POOL/mindeye_outputs/smoke_base/smoke_base_all_priorout.pt
