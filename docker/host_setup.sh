#!/usr/bin/env bash
# kdr bare-host environment setup.
#
# The normal kdr run uses the `ghcr.io/lucaspirola/kdr:latest` image, whose
# Dockerfile bakes in the full dependency environment. This script reproduces
# that environment build for a BARE host — a freshly provisioned GPU instance
# running only a CUDA base image (e.g. `nvidia/cuda:13.0.x-cudnn-devel`), with
# no pip, no build prerequisites, and no Python packages.
#
# It is the single source of truth for the bare-host path, so that path is no
# longer hand-rolled (which is how Phase 7.2's first run lost ~15 min to two
# missing prereqs: `python3-dev` → `Python.h not found`, and `wheel` → kernel
# build metadata failure). Keep this in sync with docker/Dockerfile.
#
# Usage (from the kdr repo root, after `git clone`):
#     bash docker/host_setup.sh && bash docker/bootstrap.sh
#
# Idempotent: re-running is safe. Inside the kdr image every step is already
# satisfied, so running it there is a fast no-op.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PIP_NO_CACHE_DIR=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${REPO_ROOT}/docker/requirements.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
    echo "ERROR: ${REQ_FILE} not found — run this from a kdr checkout." >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. APT build prerequisites.
#    python3-dev provides Python.h (required to compile causal-conv1d /
#    flash-linear-attention); ninja-build + build-essential provide the
#    compiler toolchain; git + curl are needed by bootstrap.sh. Mirrors the
#    Dockerfile's apt-get step.
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [1/5] apt build prerequisites"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-dev git curl ca-certificates build-essential ninja-build
# bootstrap.sh invokes the bare `python` command; the CUDA base image only
# ships `python3`. The Dockerfile creates this symlink — mirror it here.
ln -sf /usr/bin/python3 /usr/bin/python

# ─────────────────────────────────────────────────────────────────────────────
# 2. pip — Ubuntu 24.04 base images ship Python without pip.
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [2/5] pip"
if ! python3 -m pip --version >/dev/null 2>&1; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages --quiet
fi
# wheel + setuptools + packaging MUST be present before any
# `--no-build-isolation` source build (step 5), or metadata generation fails
# with `ModuleNotFoundError: No module named 'wheel'`.
python3 -m pip install -q --upgrade pip wheel setuptools packaging

# ─────────────────────────────────────────────────────────────────────────────
# 3. torch from the cu130 wheel index (matches the CUDA 13.0 base; cu130
#    wheels carry native sm_100 / Blackwell B200 support). Captured as a
#    constraint so transitive deps cannot pull a different build variant.
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [3/5] torch (cu130)"
python3 -m pip install -q --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0"
python3 -m pip freeze | grep '^torch==' > /tmp/torch-constraint.txt

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pinned dependency set.
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [4/5] docker/requirements.txt"
python3 -m pip install -q -r "${REQ_FILE}" -c /tmp/torch-constraint.txt

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compiled fast-path kernels — built against the already-installed torch
#    (--no-build-isolation), so they need steps 2-4 done first.
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [5/5] compiled kernels (causal-conv1d, flash-linear-attention)"
python3 -m pip install -q --no-build-isolation -c /tmp/torch-constraint.txt \
    "causal-conv1d>=1.4.0" \
    "flash-linear-attention>=0.1.2"

echo ">>> host_setup.sh complete — environment ready for docker/bootstrap.sh"
