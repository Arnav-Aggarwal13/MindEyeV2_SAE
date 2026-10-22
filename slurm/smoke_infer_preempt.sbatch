#!/bin/bash
#SBATCH -J mindeye-smoke-pre
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh
mkdir -p $OUT_ROOT
BIGG=$(ls $BIGG_SAFETENSORS | head -1)
source fmri/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
cd src
# Override with: sbatch --export=ALL,MAX_IMAGES=20 slurm/smoke_infer.sbatch
python -u interp/recon_infer.py --data_path $NSD_DATA --bigg_safetensors "$BIGG" \
  --out_dir $OUT_ROOT --out_name smoke_base --max_images ${MAX_IMAGES:-10}
