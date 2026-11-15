# =============================================================================
# tokenizer-v2 runtime image for a SLURM GPU cluster (developed on MareNostrum 5, NVIDIA H100)
# -----------------------------------------------------------------------------
# Built with Docker anywhere, converted to a Singularity .sif on the cluster,
# which cannot run Docker (no root, no daemon) and provides singularity/4.1.5.
#
# WHY A PLAIN UBUNTU BASE AND NOT nvidia/cuda:
#   The torch wheels already bundle the entire CUDA 12.6 userspace
#   (nvidia-cublas-cu12, nvidia-cudnn-cu12, ... — see requirements-linux.txt),
#   and `singularity exec --nv` injects the *host* driver libraries at runtime.
#   An nvidia/cuda base would add ~5 GB of a second, unused CUDA toolkit and a
#   real risk of the two userspaces shadowing each other. We only need nvcc if
#   we ever compile a custom CUDA extension — this project is pure PyTorch.
#
#   NEVER put an NVIDIA *driver* in the image: --nv injects the host's.
#
# CUDA PIN: the cluster's GPU nodes run driver 535.86.10 => "CUDA Version: 12.2".
#   cu126 works on it via CUDA minor-version compatibility (needs >= 525.60.13).
#   Do not bump to cu128+ without re-running the GPU smoke test.
# =============================================================================
FROM ubuntu:24.04

# Ubuntu 24.04 ships Python 3.12 natively — matches requires-python = ">=3.12".
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A venv sidesteps Ubuntu 24.04's PEP 668 "externally-managed-environment"
# refusal and keeps our packages away from the distro's python3.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

# Fully-pinned lock (uv pip compile) => byte-identical rebuilds.
# The +cu126 local version only exists on the PyTorch index, hence the flag.
COPY requirements-linux.txt /tmp/requirements-linux.txt
# isosplit6 ships no Python 3.12 wheel: it is compiled here, and the compiler is removed
# afterwards, so the runtime image stays compiler-free.
RUN apt-get update && apt-get install -y --no-install-recommends g++ python3.12-dev && \
    pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r /tmp/requirements-linux.txt && \
    rm /tmp/requirements-linux.txt && \
    apt-get purge -y g++ python3.12-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Inside Singularity, $HOME is bind-mounted from the host, so a stray
# ~/.local/lib/python3.12/site-packages on GPFS would silently shadow the
# image's packages. Refuse to read user-site at all.
ENV PYTHONNOUSERSITE=1

# Fail the build rather than ship a broken image (CPU-only check; the GPU half
# is verified on a compute node by cluster/container/smoke_gpu.slurm).
RUN python -c "import torch, numpy, scipy, pandas, neo, nixio, sklearn, wandb, dotenv, isosplit6; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| numpy', numpy.__version__)"

# Code and data stay OUTSIDE the image and are bind-mounted at runtime.
WORKDIR /workspace

CMD ["/bin/bash"]
