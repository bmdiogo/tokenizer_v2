#!/usr/bin/env bash
# Step 1 (any machine with Docker): build the image and export it as a docker-archive.
#
#   bash cluster/container/build_image.sh
#
# Produces cluster/container/<image>_<tag>.tar (+ .sha256). An archive, not a registry push: the
# cluster's login nodes have no outbound internet, so `singularity pull docker://...` cannot work.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./image.env

docker info >/dev/null 2>&1 || { echo "ERROR: the Docker daemon is not reachable." >&2; exit 1; }
echo ">> building ${IMAGE_NAME}:${IMAGE_TAG} (linux/amd64)"
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo ">> exporting ${TAR_NAME} (~3.7 GB)"
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${TAR_NAME}"
sha256sum "${TAR_NAME}" > "${TAR_NAME}.sha256"
echo ">> done: $(pwd)/${TAR_NAME} ($(du -h "${TAR_NAME}" | cut -f1)); next: upload it and sbatch build_sif.slurm"
