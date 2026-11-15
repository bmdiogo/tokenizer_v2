#!/usr/bin/env bash
# Run the same image locally on CPU (import checks, the test suite, small runs).
#
#   bash cluster/container/run_local.sh                     # interactive shell
#   bash cluster/container/run_local.sh python -m pytest -q
#   DATA_HOST=/path/to/data bash cluster/container/run_local.sh python -m app.precompute <recording> ...
#
# The working tree is mounted at /workspace (edits apply without a rebuild); DATA_HOST, if set, at /data.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./image.env
REPO_ROOT="$(cd ../.. && pwd)"
REPO_ROOT_HOST="$(cd ../.. && { pwd -W 2>/dev/null || pwd; })"     # Git Bash: C:/... for docker.exe

docker info >/dev/null 2>&1 || { echo "ERROR: the Docker daemon is not reachable." >&2; exit 1; }
docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1 \
    || { echo "ERROR: image ${IMAGE_NAME}:${IMAGE_TAG} not built: run build_image.sh." >&2; exit 1; }
export MSYS_NO_PATHCONV=1                                            # Git Bash: keep /workspace as is

# docker --env-file keeps inline comments, python-dotenv strips them: pass a comment-free copy of .env
ENV_ARG=()
CLEAN_ENV=""
if [[ -f "${REPO_ROOT}/.env" ]]; then
    CLEAN_ENV="$(mktemp)"
    sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//' "${REPO_ROOT}/.env" > "${CLEAN_ENV}"
    ENV_ARG=(--env-file "$(cygpath -w "${CLEAN_ENV}" 2>/dev/null || echo "${CLEAN_ENV}")")
fi
TTY_ARG=(-i)
[[ -t 0 ]] && TTY_ARG=(-i -t)
DATA_ARG=()
[[ -n "${DATA_HOST:-}" ]] && DATA_ARG=(-v "${DATA_HOST}:/data")

status=0
docker run --rm "${TTY_ARG[@]}" -v "${REPO_ROOT_HOST}:/workspace" "${DATA_ARG[@]}" -w /workspace \
    -e PYTHONPATH=/workspace -e WANDB_MODE=offline "${ENV_ARG[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" "${@:-bash}" || status=$?
[[ -n "${CLEAN_ENV}" ]] && rm -f "${CLEAN_ENV}"
exit ${status}
