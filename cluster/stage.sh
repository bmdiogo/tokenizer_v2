#!/usr/bin/env bash
# Submit one stage of the experiment plan: its training runs (scripts/stage.py), then scripts.compare on them.
#
#   AFTER=<gate job> SCORE_AFTER=<T7 scorer job> cluster/stage.sh s5   # runs wait for the gate, the scorer for T7
#   REF=s5-c3 cluster/stage.sh s6                                       # the 125 Hz reference = the S5 winner
#   REF=s5-c3 cluster/stage.sh s7
#   cluster/stage.sh s8 --finalists ref s6-313hz-lr1                    # 5 seeds x 200 k, 3 chained 24 h legs; scorer = S9
#   cluster/stage.sh s12                                                # frozen-encoder ŷ, 3 seeds x 50 k, scored once
#
# The run list is committed before anything is submitted, so the draw is fixed before any number is
# seen. Settings build on the local .env (s6 reads LR) and on the cluster's .env (everything else).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$here/cluster.env"
: "${CLUSTER_REPO:?set CLUSTER_REPO in cluster/cluster.env}"
stage="${1:?usage: cluster/stage.sh s5|s6|s7|s8|s12 [--finalists ...]}"
shift

cd "$here/.."
lines="$(python -m scripts.stage "$stage" "$@")"
git add "experiments/${stage}_arms.json"
git commit -q -m "Stage ${stage}: the runs, fixed before they start" -- "experiments/${stage}_arms.json" || true
git push "$CLUSTER_ALIAS" "HEAD:${CLUSTER_BRANCH}"

if [ "$stage" = s8 ]; then legs=3; time=24:00:00; score="--stage s9 --steps 50000 100000 150000 200000"
elif [ "$stage" = s12 ]; then legs=2; time=24:00:00
    score="--stage s12 --steps 50000 --groups p2_late p3_late p2_final p3_final p4_unseen"
else legs=1; time="${TIME:-08:00:00}"; score="--stage ${stage}"; fi
remote="set -e; source cluster/cluster.env; mkdir -p logs/slurm; ids=''"
names=""
while IFS=$'\t' read -r name extra; do
    names="$names $name"
    dep="${AFTER:+--dependency=afterok:$AFTER}"
    for leg in $(seq 1 "$legs"); do                       # a leg after a timeout resumes from last.pt
        remote="$remote; J=\$(EXP='$name' EXTRA='$extra' sbatch --parsable --job-name='tv2-$name' --time=$time $dep cluster/train.slurm); echo '$name' leg $leg \$J"
        dep="--dependency=afterany:\$J"
    done
    remote="$remote; ids=\$ids:\$J"
done <<< "$lines"
cmd="python -m scripts.compare $score --runs$names --jobs 4${REF:+ --ref $REF}"
remote="$remote; S=\$(CMD='$cmd' sbatch --parsable --job-name='tv2-${stage}-score' --time=${SCORE_TIME:-16:00:00} --dependency=afterany\$ids${SCORE_AFTER:+,afterok:$SCORE_AFTER} cluster/tool.slurm); echo score \$S"
remote="$remote; squeue -u \$USER -o '%.10i %.20j %.8T %.20R %E'"
ssh "$CLUSTER_ALIAS" "cd '$CLUSTER_REPO' && $remote"
echo "submitted ${stage} from $(git rev-parse --short HEAD)"
