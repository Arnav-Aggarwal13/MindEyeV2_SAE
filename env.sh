# Paths for MindEyeV2_SAE. Usage: source env.sh (from the repo root)
export POOL=$HOME/orcd/pool
export SCR=$HOME/orcd/scratch/mindeye
export NSD_DATA=$POOL/mindeye_data                       # flat HF dataset dir (betas, wds/, evals/, bigG_to_L, ...), read from pool
export CKPT_DIR=$SCR/ckpts                               # scratch copies: unclip6 ckpt, bigG safetensors, <model_name>/last.pth
export BIGG_SAFETENSORS=$CKPT_DIR/open_clip_model.safetensors
export OUT_ROOT=$SCR/outputs
export HF_HOME=$POOL/mindeye_hf                          # microsoft/git-large-coco
export HF_HUB_OFFLINE=1
export TORCH_HOME=$POOL/mindeye_hf/torch
