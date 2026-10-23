#!/bin/bash
#SBATCH -J mindeye-umap
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
cd src
B=$OUT_ROOT/base
python -u interp/eval_recons.py --sections umap --tag umap --data_path $NSD_DATA \
  --clip_target $B/base_clip_target.pt --prior_out $B/base_all_prior_out.pt --clipvoxels $B/base_all_clipvoxels.pt \
  --out_dir $OUT_ROOT --out_name ${EVAL_NAME:-eval_base_recons}
