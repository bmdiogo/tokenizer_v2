#!/bin/bash
# Run any command inside the container (cache build, pre-flight tests, evaluation, figure data).
#
#   CMD='python -m utils.cache --patients P2 --workers 16' sbatch --time=12:00:00 cluster/tool.slurm
#   CMD='python -m scripts.evaluate runs/t7-s1/step_015000.pt --group p2_late' sbatch cluster/tool.slurm
#SBATCH --job-name=tv2-tool
#SBATCH --partition=acc
#SBATCH --qos=acc_resb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm logs/module
: "${CMD:?set CMD='python -m ...'}"
source cluster/cluster.env
source cluster/container/image.env
module load "${SINGULARITY_MODULE}" > "logs/module/${SLURM_JOB_ID}.log" 2>&1
echo "$(date +%T) $(hostname) ${CMD}"
srun singularity exec --nv --cleanenv \
    --bind "${CLUSTER_BASE}:${CLUSTER_BASE}" --bind "${CLUSTER_REPO}:/workspace" --pwd /workspace \
    --env PYTHONPATH=/workspace,PYTHONUNBUFFERED=1,WANDB_MODE=offline,WANDB_DIR=/workspace/logs \
    "${CLUSTER_CONTAINERS}/${SIF_NAME}" bash -c "${CMD}"
echo "$(date +%T) done"
