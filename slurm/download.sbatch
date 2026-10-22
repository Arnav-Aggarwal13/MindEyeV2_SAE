#!/bin/bash
#SBATCH -J mindeye-download
#SBATCH -p mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
# Download the MindEye2 HF dataset into pool (resumable; rerun if it times out).
cd /home/arnav813/orcd/pool/mindeye_data || exit 1
echo "host: $(hostname)"
curl -sI -m 15 https://huggingface.co | head -1 || { echo "NO INTERNET"; exit 1; }
/home/arnav813/MindEyeV2_SAE/fmri/bin/python -I /home/arnav813/MindEyeV2_SAE/scripts/download_data.py
