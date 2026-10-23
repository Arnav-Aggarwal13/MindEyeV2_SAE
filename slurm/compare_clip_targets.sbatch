#!/bin/bash
#SBATCH -J mindeye-cmp-targets
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
cd src && python -u interp/compare_clip_targets.py --base_dir $OUT_ROOT/base
