#!/bin/bash
#SBATCH -J mindeye-prefetch
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
unset HF_HUB_OFFLINE   # this job needs the internet
python -I -m py_compile src/interp/eval_recons.py src/interp/prefetch_eval_models.py && echo compiled
cd src && python -u interp/prefetch_eval_models.py
