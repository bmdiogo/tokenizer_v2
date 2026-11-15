# Running on a SLURM cluster

Every training run and every test-set evaluation of the thesis ran on MareNostrum 5 (BSC), one
NVIDIA H100 per job, inside a Singularity image built from `container/Dockerfile`. The scripts
assume SLURM, Singularity and a git checkout of this repository on the cluster's shared filesystem.
Partition and QoS names in the `#SBATCH` headers are MareNostrum 5's; adapt them to your site.

| file | runs on | does |
|---|---|---|
| `cluster.env.example` | both | account, ssh alias and paths; copy to `cluster.env` (gitignored) locally **and** in the cluster checkout |
| `container/Dockerfile` | Docker host | Ubuntu 24.04, Python 3.12, the pinned lock `requirements-linux.txt` (torch 2.8.0+cu126) |
| `container/build_image.sh` | Docker host | `docker build` → `docker save` → `.tar` + `.sha256` |
| `container/build_sif.slurm` | cluster | converts the uploaded archive to a `.sif` on a compute node |
| `container/smoke_gpu.slurm` | cluster | checks the GPU is visible in the container and runs the codec once |
| `container/run_local.sh` | Docker host | the same image on CPU, with the working tree mounted |
| `train.slurm` | cluster | one training run (`EXP`, `EXTRA`); copies the training cache to node-local scratch |
| `tool.slurm` | cluster | any one command in the container (`CMD`): cache, pre-flight tests, evaluation |
| `submit.sh` | local | `git push` to the cluster checkout, then `sbatch train.slurm` |
| `stage.sh` | local | one stage of the plan: commits its run list, submits the runs and the scorer that waits for them |

## Setup, once

```bash
cp cluster/cluster.env.example cluster/cluster.env        # fill it in; copy it to the cluster checkout too
git remote add hpc <alias>:<CLUSTER_REPO>                  # the remote's name = CLUSTER_ALIAS
bash cluster/container/build_image.sh                      # needs Docker; ~10 min, a 3.7 GB archive
scp cluster/container/tokenizer-v2_0.2.0.tar* <user>@<transfer node>:<CLUSTER_CONTAINERS>/
ssh hpc 'cd <CLUSTER_REPO> && sbatch cluster/container/build_sif.slurm'    # log ends with SIF_BUILD_OK
ssh hpc 'cd <CLUSTER_REPO> && sbatch cluster/container/smoke_gpu.slurm'    # log ends with ALL CHECKS PASSED
```

The cluster checkout needs its own `.env` (gitignored) with the cluster's `CACHE_DIR`, `RUNS_DIR`
and `WANDB_MODE=offline`: compute nodes have no internet, so W&B runs offline and is synced
afterwards with `wandb sync`.

## Everyday use

```bash
EXP=t7-s1 EXTRA='SEED=1 DATA_SEED=11 MAX_STEPS=15000' cluster/submit.sh
ssh hpc "cd <CLUSTER_REPO> && CMD='python -m scripts.preflight.t3_threshold' sbatch cluster/tool.slurm"
REF=s5-c8 cluster/stage.sh s7
```

Overrides go in `EXTRA` as `KEY=VALUE`, **never** inside `sbatch --export=...`: sbatch splits that
list on commas, so `STRIDES=8,5,3,2,2` would be cut silently. Re-submitting a run name resumes it
from `runs/<name>/last.pt`; a 200 k-step run is three chained 24 h legs.

## Notes on the image

- **No `nvidia/cuda` base.** The torch wheels bundle the CUDA 12.6 userspace and `--nv` injects the
  host driver, so a CUDA base image would only add a second, unused toolkit (~5 GB).
- **CUDA pin.** The nodes run driver 535.86.10 (CUDA 12.2). A cu126 build runs on it through CUDA
  minor-version compatibility (driver ≥ 525.60.13); re-run `smoke_gpu.slurm` after changing the pin.
- **Archive, not registry.** Login nodes have no outbound internet, so the image travels as a
  `docker save` archive and is converted with `singularity build x.sif docker-archive://x.tar`, on a
  compute node (the login node's watchdog kills mksquashfs). User namespaces are disabled, so a
  `.def` build cannot work there.
- **`--cleanenv`** keeps a `module load`'s `PYTHONPATH` from shadowing the image's venv, and the
  image sets `PYTHONNOUSERSITE=1` because `$HOME` is bind-mounted from the host.
- **Identity bind** of `CLUSTER_BASE`: every absolute path in the cluster `.env` resolves unchanged
  inside the container. Code and data stay outside the image, so a code change is a `git push`;
  only a dependency change needs a new image (bump `IMAGE_TAG` in `container/image.env`).
- **isosplit6** (the clusterer of the offline sorter) has no Python 3.12 wheel. It is compiled at
  build time and the compiler is removed afterwards.
