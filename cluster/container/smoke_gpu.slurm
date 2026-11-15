#!/bin/bash
# Step 3 (cluster): check that the container sees the GPU and runs the codec before trusting it with a run.
#
#   sbatch cluster/container/smoke_gpu.slurm        # from the repository root
#SBATCH --job-name=tv2-smoke
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm logs/module
source cluster/cluster.env
source cluster/container/image.env
module load "${SINGULARITY_MODULE}" > "logs/module/smoke_${SLURM_JOB_ID}.log" 2>&1
SIF="${CLUSTER_CONTAINERS}/${SIF_NAME}"
[[ -f "${SIF}" ]] || { echo "ERROR: ${SIF} missing: run build_sif.slurm first." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --nv injects the host driver; --cleanenv keeps a module's PYTHONPATH from shadowing the image's venv
singularity exec --nv --cleanenv \
    --bind "${CLUSTER_BASE}:${CLUSTER_BASE}" --bind "${CLUSTER_REPO}:/workspace" \
    --pwd /workspace --env PYTHONPATH=/workspace \
    "${SIF}" python - <<'PY'
import platform

import torch

print("python :", platform.python_version())
print("torch  :", torch.__version__, "| CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "GPU not visible inside the container (--nv / driver mismatch)"
print("device :", torch.cuda.get_device_name(0), "| sm_%d%d" % torch.cuda.get_device_capability(0))

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("cuBLAS :", torch.matmul(a, a).float().mean().item())

from models.codec import Codec                       # the repository imports from the bind mount

model = Codec().cuda()
with torch.autocast("cuda", torch.bfloat16):
    x_hat, y_hat, z = model(torch.randn(1, 96, 7680, device="cuda"))
print("codec  :", model.describe(), "| z", tuple(z.shape))
print("ALL CHECKS PASSED")
PY
