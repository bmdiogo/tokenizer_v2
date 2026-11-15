#!/bin/bash
# Step 2 (cluster): convert the uploaded docker-archive into a .sif, on a compute node.
#
#   scp cluster/container/tokenizer-v2_0.2.0.tar* <transfer node>:$CLUSTER_CONTAINERS/
#   sbatch cluster/container/build_sif.slurm        # from the repository root
#
# A job and not a login-node command: expanding 3.7 GB -> 10.8 GB and running mksquashfs was killed
# by the login node's watchdog. No GPU is needed.
#SBATCH --job-name=tv2-sif-build
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm logs/module
source cluster/cluster.env
source cluster/container/image.env
module load "${SINGULARITY_MODULE}" > "logs/module/sif_${SLURM_JOB_ID}.log" 2>&1

cd "${CLUSTER_CONTAINERS}"
[[ -f "${TAR_NAME}" ]] || { echo "ERROR: ${TAR_NAME} not found: upload it first." >&2; exit 1; }
[[ -f "${TAR_NAME}.sha256" ]] && sha256sum -c "${TAR_NAME}.sha256"

# node-local scratch for mksquashfs (faster than the shared filesystem, outside its quota)
export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/sing_tmp_${SLURM_JOB_ID}"
export SINGULARITY_CACHEDIR="${TMPDIR:-/tmp}/sing_cache_${SLURM_JOB_ID}"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"' EXIT

rm -f "${SIF_NAME}"
singularity build "${SIF_NAME}" "docker-archive://${TAR_NAME}"
ls -lh "${SIF_NAME}"
echo "SIF_BUILD_OK"
