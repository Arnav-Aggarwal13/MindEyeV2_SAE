#!/bin/bash
#SBATCH -J mindeye-copy
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
POOL=$HOME/orcd/pool; DEST=$HOME/orcd/scratch/mindeye/ckpts
BIGG=$(ls $POOL/hf_cache/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/*/open_clip_model.safetensors | head -1)
mkdir -p $DEST
for f in $POOL/mindeye_data/unclip6_epoch0_step110000.ckpt "$BIGG" $POOL/mindeye_data/train_logs/final_subj01_pretrained_40sess_24bs/last.pth; do
  echo "$(date +%T) copying $f"
  cp -L "$f" $DEST/
  [ "$(stat -L -c %s "$f")" = "$(stat -c %s $DEST/$(basename "$f"))" ] && echo "ok $(basename "$f")" || { echo "SIZE MISMATCH $f"; exit 1; }
done
ls -la $DEST; echo DONE
