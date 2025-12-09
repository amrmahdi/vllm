# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/vllm-build-base:cuda12.9-ubuntu22.04-py3.12

# Pre-built build base image with Python, CUDA devel tools, uv, and PyTorch
# Built from docker/Dockerfile.build-base - see .buildkite/pipeline-base-image.yaml
# Fallback to nvidia/cuda devel image if not available
ARG BUILD_BASE_IMAGE=public.ecr.aws/q9t5s3a7/vllm-build-base:cuda12.9-ubuntu22.04-py3.12
# Pre-built runtime base image with Python, minimal CUDA tools, PyTorch, and FlashInfer
# Built from docker/Dockerfile.runtime-base - see .buildkite/pipeline-base-image.yaml
ARG FINAL_BASE_IMAGE=public.ecr.aws/q9t5s3a7/vllm-runtime-base:cuda12.9-ubuntu22.04-py3.12

# PIP supports fetching the packages from custom indexes, allowing third-party
# to host the packages in private mirrors. The PIP_INDEX_URL and
# PIP_EXTRA_INDEX_URL are standard PIP environment variables to override the
# default indexes. By letting them empty by default, PIP will use its default
# indexes if the build process doesn't override the indexes.
#
# Uv uses different variables. We set them by default to the same values as
# PIP, but they can be overridden.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# PyTorch provides its own indexes for standard and nightly builds
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl

# PIP supports multiple authentication schemes, including keyring
# By parameterizing the PIP_KEYRING_PROVIDER variable and setting it to
# disabled by default, we allow third-party to use keyring authentication for
# their private Python indexes, while not changing the default behavior which
# is no authentication.
#
# Reference: https://pip.pypa.io/en/stable/topics/authentication/#keyring-support
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# Flag enables built-in KV-connector dependency libs into docker images
ARG INSTALL_KV_CONNECTORS=false

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM ${BUILD_BASE_IMAGE} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false
ENV DEBIAN_FRONTEND=noninteractive

# Base image already has: Python, uv, venv, PyTorch, gcc-10, ccache, build tools
# See docker/Dockerfile.build-base for what's pre-installed

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Activate virtual environment and add uv to PATH
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# uv settings (already set in base but redeclare for clarity)
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Copy requirements files (needed by setup.py, already installed in base image)
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt

# cuda arch list used by torch
ARG TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
#################### BASE BUILD IMAGE ####################

#################### CSRC BUILD IMAGE ####################
FROM base AS csrc-build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# install build dependencies
COPY requirements/build.txt requirements/build.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python3 -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

WORKDIR /workspace

COPY pyproject.toml setup.py CMakeLists.txt ./
COPY cmake cmake/
COPY csrc csrc/
COPY vllm/envs.py vllm/envs.py
COPY vllm/__init__.py vllm/__init__.py

# max jobs used by Ninja to build extensions
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}
# number of threads used by nvcc
ARG NVCC_THREADS=8
ENV NVCC_THREADS=${NVCC_THREADS}

ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED=""
ARG VLLM_MERGE_BASE_COMMIT=""
ARG VLLM_MAIN_CUDA_VERSION=""

# Use dummy version for csrc-build wheel (only .so files are extracted, version doesn't matter)
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+csrc.build"

# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && if [ ! -z ${SCCACHE_ENDPOINT} ] ; then export SCCACHE_ENDPOINT=${SCCACHE_ENDPOINT} ; fi \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" \
        && export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}" \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi
#################### CSRC BUILD IMAGE ####################

#################### EXTENSIONS BUILD IMAGE ####################
# Build DeepGEMM, pplx-kernels, DeepEP - runs in PARALLEL with csrc-build
# This stage is independent and doesn't affect csrc cache
FROM base AS extensions-build
ARG CUDA_VERSION

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Build DeepGEMM wheel
ARG DEEPGEMM_GIT_REF
COPY tools/install_deepgemm.sh /tmp/install_deepgemm.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /tmp/deepgemm/dist && \
    VLLM_DOCKER_BUILD_CONTEXT=1 TORCH_CUDA_ARCH_LIST="9.0a 10.0a" /tmp/install_deepgemm.sh \
        --cuda-version "${CUDA_VERSION}" \
        ${DEEPGEMM_GIT_REF:+--ref "$DEEPGEMM_GIT_REF"} \
        --wheel-dir /tmp/deepgemm/dist || \
    echo "DeepGEMM build skipped (CUDA version requirement not met)"

# Ensure the wheel dir exists so COPY won't fail when DeepGEMM is skipped
RUN mkdir -p /tmp/deepgemm/dist && touch /tmp/deepgemm/dist/.deepgemm_skipped

# Build pplx-kernels and DeepEP wheels
COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh
ARG PPLX_COMMIT_HASH
ARG DEEPEP_COMMIT_HASH
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /tmp/ep_kernels_workspace/dist && \
    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \
    /tmp/install_python_libraries.sh \
        --workspace /tmp/ep_kernels_workspace \
        --mode wheel \
        ${PPLX_COMMIT_HASH:+--pplx-ref "$PPLX_COMMIT_HASH"} \
        ${DEEPEP_COMMIT_HASH:+--deepep-ref "$DEEPEP_COMMIT_HASH"} && \
    find /tmp/ep_kernels_workspace/nvshmem -name '*.a' -delete
#################### EXTENSIONS BUILD IMAGE ####################

#################### COMPILED WHEELS (minimal cache layer) ####################
# FROM scratch creates a tiny layer with ONLY the compiled artifacts
# This makes cache extraction fast (~150MB vs multi-GB build layers)
# Contains: csrc wheel + extension wheels (deepgemm, pplx, deepep) + nvshmem
FROM scratch AS compiled-wheels
COPY --from=csrc-build /workspace/dist/*.whl /csrc/
COPY --from=extensions-build /tmp/deepgemm/dist/*.whl /extensions/
COPY --from=extensions-build /tmp/ep_kernels_workspace/dist/*.whl /extensions/
COPY --from=extensions-build /tmp/ep_kernels_workspace/nvshmem /nvshmem/
#################### COMPILED WHEELS ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# install build dependencies
COPY requirements/build.txt requirements/build.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python3 -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

WORKDIR /workspace

# Copy all pre-built artifacts from minimal cache layer
# This is FAST because compiled-wheels is a tiny FROM scratch layer (~200MB)
COPY --from=compiled-wheels /csrc /precompiled-wheels
COPY --from=compiled-wheels /extensions /tmp/extension-wheels
COPY --from=compiled-wheels /nvshmem /tmp/ep_kernels_workspace/nvshmem

COPY . .

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}

# Skip adding +precompiled suffix to version (preserves git-derived version)
ENV VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "${vllm_target_device}" = "cuda" ]; then \
        export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl); \
    fi && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# Organize wheels into expected locations for later stages
RUN mkdir -p /tmp/deepgemm/dist /tmp/ep_kernels_workspace/dist && \
    cp /tmp/extension-wheels/deep_gemm*.whl /tmp/deepgemm/dist/ 2>/dev/null || touch /tmp/deepgemm/dist/.deepgemm_skipped && \
    cp /tmp/extension-wheels/pplx_kernels*.whl /tmp/ep_kernels_workspace/dist/ && \
    cp /tmp/extension-wheels/deep_ep*.whl /tmp/ep_kernels_workspace/dist/

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=500
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi
#################### EXTENSION Build IMAGE ####################

#################### DEV IMAGE ####################
FROM base AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Install libnuma-dev, required by fastsafetensors (fixes #20384)
RUN apt-get update && apt-get install -y --no-install-recommends libnuma-dev && rm -rf /var/lib/apt/lists/*
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test.txt requirements/test.txt
COPY requirements/dev.txt requirements/dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python3 -r requirements/dev.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
#################### DEV IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG INSTALL_KV_CONNECTORS=false
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

SHELL ["/bin/bash", "-c"]

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Python, CUDA tools, uv, and FlashInfer are pre-installed in FINAL_BASE_IMAGE
# See docker/Dockerfile.base for what's included

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install vllm wheel first, so that torch etc will be installed.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .

RUN --mount=type=cache,target=/root/.cache/uv \
. /etc/environment && \
uv pip list

# Install deepgemm wheel that has been built in the `build` stage
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=build,source=/tmp/deepgemm/dist,target=/tmp/deepgemm/dist,ro \
    sh -c 'if ls /tmp/deepgemm/dist/*.whl >/dev/null 2>&1; then \
              uv pip install --system /tmp/deepgemm/dist/*.whl; \
           else \
              echo "No DeepGEMM wheels to install; skipping."; \
           fi'

# Pytorch now installs NVSHMEM, setting LD_LIBRARY_PATH (https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/.ci/manywheel/build_cuda.sh#L141C14-L141C36)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install EP kernels wheels (pplx-kernels and DeepEP) that have been built in the `build` stage
RUN --mount=type=bind,from=build,src=/tmp/ep_kernels_workspace/dist,target=/vllm-workspace/ep_kernels/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ep_kernels/dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# gdrcopy is pre-installed in runtime-base image

# CUDA image changed from /usr/local/nvidia to /usr/local/cuda in 12.8 but will
# return to /usr/local/nvidia in 13.0 to allow container providers to mount drivers
# consistently from the host (see https://github.com/vllm-project/vllm/issues/18859).
# Until then, add /usr/local/nvidia/lib64 before the image cuda path to allow override.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

#################### vLLM installation IMAGE ####################

#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

ARG PYTHON_VERSION

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y git

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        uv pip install --system -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e tests/vllm_test_utils

# enable fast downloads from hf (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Copy in the v1 package for testing (it isn't distributed yet)
COPY vllm/v1 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/vllm/v1

# Source code is used in the `python_only_compile.sh` test
# We hide it inside `src/` so that this source code
# will not be imported by other tests
RUN mkdir src
RUN mv vllm src/vllm
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install kv_connectors if requested (openai deps are pre-installed in runtime-base)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements/kv_connectors.txt,target=/tmp/kv_connectors.txt,ro \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        uv pip install --system -r /tmp/kv_connectors.txt; \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["vllm", "serve"]
#################### OPENAI API SERVER ####################
