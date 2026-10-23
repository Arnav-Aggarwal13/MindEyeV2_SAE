#!/bin/bash
#SBATCH -J mindeye-dl-recons
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh && source fmri/bin/activate
unset HF_HUB_OFFLINE
curl -s -m 15 -o /dev/null -w "huggingface.co reachable: HTTP %{http_code}\n" https://huggingface.co || echo "NO INTERNET from this node"
python -I - <<'PY'
import os
from huggingface_hub import hf_hub_download
D = os.path.expanduser("~/orcd/pool/mindeye_data")
f = "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_recons.pt"
p = hf_hub_download("pscotti/mindeyev2", filename=f, repo_type="dataset", revision="main", local_dir=D)
print(p, os.path.getsize(p))
PY
