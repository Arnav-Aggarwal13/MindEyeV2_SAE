#!/bin/bash
#SBATCH -J mindeye-embed
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
cd src
B=$OUT_ROOT/base
python -u interp/embed_metrics.py \
  --brain prior_out=$B/base_all_prior_out.pt clip_voxels=$B/base_all_clipvoxels.pt \
  --image clip_target=$B/base_clip_target.pt \
  --ref prior_out --out_dir $B/embed_metrics --tag base
