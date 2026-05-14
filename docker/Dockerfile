# kdr portable runtime image — companion to docker/bootstrap.sh.
#
# Targets vast.ai / RunPod / Lambda / any GPU host that supports `docker run --gpus all`.
#
# Build context: the kdr repo root (so docker/requirements.txt + docker/bootstrap.sh
# are addressable from `docker/...` paths).
# Build:   docker build -f docker/Dockerfile -t kdr:dev .
# Push:    handled by .github/workflows/docker-build.yml → ghcr.io/lucaspirola/kdr
#
# Image carries dependencies only. Code is git-cloned at container start by
# docker/bootstrap.sh; model snapshot is downloaded into the persistent /cache
# volume.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/hub \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# Ubuntu 24.04 ships Python 3.12. git for runtime code clone; build tooling
# for compiled wheels (flash-linear-attention and causal-conv1d build CUDA
# kernels — need ninja + nvcc; the cuda:13.0.3-devel base provides nvcc with
# sm_100 codegen support).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools packaging

# torch from the cu130 wheel index (matches CUDA 13.0 base; cu130 wheels
# add native sm_100 / Blackwell B200 support). Capture the installed wheel
# as a constraint so transitive deps cannot pull a different build variant
# via the default PyPI index.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.11.0" \
    && pip freeze | grep '^torch==' > /tmp/torch-constraint.txt

# Bulk install of pinned deps. The constraint file locks torch to the exact
# cu130 wheel above.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -c /tmp/torch-constraint.txt

# Compiled fast-path kernels need --no-build-isolation so they can see the
# already-installed torch + cuda toolkit during their setup.py.
RUN pip install --no-build-isolation -c /tmp/torch-constraint.txt \
        "causal-conv1d>=1.4.0" \
        "flash-linear-attention>=0.1.2"

WORKDIR /workspace

COPY docker/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# Default mountpoint for persistent storage (model snapshot, ablation
# artifacts, cloned code). On vast.ai mount as `-v /workspace/cache:/cache`.
VOLUME ["/cache"]

ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]
