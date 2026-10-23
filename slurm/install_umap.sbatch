#!/bin/bash
#SBATCH -J mindeye-umap-install
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/MindEyeV2_SAE && source env.sh
# constrain every already-pinned package to its frozen version so umap-learn cannot change the working env
# (pyarrow excluded: we deliberately run 20.0.0 instead of the frozen 25.0.1; direct-URL lines dropped)
grep -vE '^pyarrow|@|^-e' requirements-working.txt > $SCR/tmp/constraints.txt
echo "--- dry run"; uv pip install --python fmri/bin/python --dry-run -c $SCR/tmp/constraints.txt umap-learn 2>&1 | tail -15
echo "--- install"; uv pip install --python fmri/bin/python -c $SCR/tmp/constraints.txt umap-learn 2>&1 | tail -15
fmri/bin/python -I -c "import umap, numba, pynndescent, numpy; print('umap', umap.__version__, 'numba', numba.__version__, 'numpy', numpy.__version__)"
