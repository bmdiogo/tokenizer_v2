#!/bin/bash
# Train inside the container. Settings come from the cluster's .env; EXTRA adds KEY=VALUE overrides.
#
#   EXP=t7-s1 EXTRA='SEED=1 DATA_SEED=11 MAX_STEPS=15000 WANDB_GROUP=t7' sbatch cluster/train.slurm
#
# EXTRA is an environment variable, never part of --export=...: sbatch splits that list on commas
# (STRIDES=8,5,3,2,2 would be cut silently). Re-submitting the same EXP resumes it from
# runs/<EXP>/last.pt. Partition and QoS are those of MareNostrum 5 (ACC, 1 x H100); adapt them.
#SBATCH --job-name=tokenizer-v2
#SBATCH --partition=acc
#SBATCH --qos=acc_resb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm logs/module
source cluster/cluster.env
source cluster/container/image.env
module load "${SINGULARITY_MODULE}" > "logs/module/${SLURM_JOB_ID}.log" 2>&1
: "${EXP:?set EXP=<run name>}"
envval() { grep -E "^$1=" .env | cut -d= -f2- | sed 's/#.*//' | xargs; }
SHARED_CACHE="$(envval CACHE_DIR)"
PATIENTS="$(envval TRAIN_PATIENTS)"                  # only the training patients are copied (never P4)

# --- copy the training recordings to node-local scratch, one copy per cache directory; jobs on the
#     same node share it (flock) and the last one to leave deletes it
LOCAL_CACHE="/scratch/tmp/tv2_${USER}_$(basename "${SHARED_CACHE}")"
mkdir -p "${LOCAL_CACHE}"
FILTER=()
for p in ${PATIENTS}; do FILTER+=(--include="/${p}/" --include="/${p}/**"); done
exec {LOCK}>"${LOCAL_CACHE}.lock"
flock "${LOCK}"
echo "$(date +%T) rsync ${PATIENTS} ${SHARED_CACHE} -> ${LOCAL_CACHE}"
rsync -a --partial "${FILTER[@]}" --exclude='/*' "${SHARED_CACHE}/" "${LOCAL_CACHE}/"
flock -u "${LOCK}"
echo "$(date +%T) cache ready ($(du -sh "${LOCAL_CACHE}" | cut -f1))"

# private /dev/shm and /tmp per job: in a shared /tmp, concurrent runs lost W&B media files
SHM_DIR="${TMPDIR:-/scratch/tmp}/tv2_shm_${SLURM_JOB_ID}"
JOB_TMP="${TMPDIR:-/scratch/tmp}/tv2_tmp_${SLURM_JOB_ID}"
mkdir -p "${SHM_DIR}" "${JOB_TMP}" && chmod 1777 "${SHM_DIR}" "${JOB_TMP}"
cleanup() {
    rm -rf "${SHM_DIR}" "${JOB_TMP}"
    local others
    others=$(squeue -h -u "${USER}" -o '%i %N' | awk -v me="${SLURM_JOB_ID}" -v h="$(hostname -s)" \
             '$1 != me && $2 == h {n++} END {print n+0}')
    [[ "${others}" == "0" ]] && rm -rf "${LOCAL_CACHE}"
    return 0
}
trap cleanup EXIT

srun singularity exec --nv --cleanenv \
    --bind "${CLUSTER_BASE}:${CLUSTER_BASE}" --bind "${CLUSTER_REPO}:/workspace" \
    --bind "${LOCAL_CACHE}:${LOCAL_CACHE}" --bind "${SHM_DIR}:/dev/shm" --bind "${JOB_TMP}:/tmp" \
    --pwd /workspace \
    --env PYTHONPATH=/workspace,PYTHONUNBUFFERED=1,WANDB_MODE=offline,WANDB_DIR=/workspace/logs \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${CLUSTER_CONTAINERS}/${SIF_NAME}" \
    python -m scripts.train --exp-name "${EXP}" CACHE_DIR="${LOCAL_CACHE}" ${EXTRA:-}
