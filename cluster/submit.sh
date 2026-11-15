#!/usr/bin/env bash
# Deploy the current commit to the cluster checkout and submit one training run.
#
#   EXP=t7-s1 EXTRA='SEED=1 DATA_SEED=11 MAX_STEPS=15000 WANDB_GROUP=t7' cluster/submit.sh
#   EXP=sanity EXTRA='MAX_STEPS=2000 EVAL_INTERVAL=250' SBATCH='--qos=acc_debug --time=02:00:00' cluster/submit.sh
#
# Settings come from the cluster's .env; EXTRA holds KEY=VALUE overrides for this run only.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$here/cluster.env"
: "${CLUSTER_REPO:?set CLUSTER_REPO in cluster/cluster.env}"
: "${EXP:?set EXP=<run name>}"

git push "$CLUSTER_ALIAS" "HEAD:${CLUSTER_BRANCH}"
ssh "$CLUSTER_ALIAS" "cd '$CLUSTER_REPO' && source cluster/cluster.env && mkdir -p logs/slurm && \
  EXP='${EXP}' EXTRA='${EXTRA:-}' sbatch ${SBATCH:-} --job-name='${EXP}' cluster/train.slurm && squeue -u \$USER"
echo "submitted ${EXP} from $(git rev-parse --short HEAD)"
