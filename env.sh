# Paths for MindEyeV2_SAE. Usage: source env.sh (from the repo root)
export POOL=$HOME/orcd/pool
export REPO=$HOME/MindEyeV2_SAE
export NSD_DATA=$POOL/mindeye_data                       # flat HF dataset dir (betas, wds/, ckpts, evals/, train_logs/)
export HF_HOME=$POOL/mindeye_hf                          # microsoft/git-large-coco
export BIGG_SAFETENSORS=$POOL/hf_cache/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/*/open_clip_model.safetensors  # shared cache, read only
export OUT_ROOT=$POOL/mindeye_outputs
export HF_HUB_OFFLINE=1
export TORCH_HOME=$POOL/mindeye_hf/torch
