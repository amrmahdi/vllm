#!/bin/bash
# Build script for a single vLLM CI base image
#
# Usage:
#   ./build-base-image.sh --dockerfile docker/Dockerfile.build-base --image-name vllm-build-base
#   ./build-base-image.sh --dockerfile docker/Dockerfile.runtime-base --image-name vllm-runtime-base
#   ./build-base-image.sh --dockerfile docker/Dockerfile.build-base --image-name vllm-build-base --no-cache
#   ./build-base-image.sh --dockerfile docker/Dockerfile.build-base --image-name vllm-build-base --dry-run
#
# Automatically disables cache for scheduled builds (weekly refresh)

set -euo pipefail

# Configuration (can be overridden via environment)
REGISTRY="${REGISTRY:-public.ecr.aws/q9t5s3a7}"
CUDA_VERSION="${CUDA_VERSION:-12.9.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

# Optional build args (passed through if set)
FLASHINFER_VERSION="${FLASHINFER_VERSION:-}"

# Derived values
CUDA_SHORT=$(echo "${CUDA_VERSION}" | cut -d. -f1,2)
TAG_SUFFIX="cuda${CUDA_SHORT}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}"

# Auto-detect: disable cache for scheduled builds (weekly invalidation)
# BUILDKITE_SOURCE values: schedule, webhook, ui, api
if [ "${BUILDKITE_SOURCE:-}" = "schedule" ]; then
    USE_CACHE=false
    echo "📅 Scheduled build detected - disabling cache for weekly refresh"
else
    USE_CACHE=true
fi

# Parse arguments
DRY_RUN=false
DOCKERFILE=""
IMAGE_NAME=""
EXTRA_BUILD_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --with-cache)
            USE_CACHE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --build-arg)
            EXTRA_BUILD_ARGS="${EXTRA_BUILD_ARGS} --build-arg $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${DOCKERFILE}" ]; then
    echo "Error: --dockerfile is required"
    echo "Usage: ./build-base-image.sh --dockerfile <path> --image-name <name>"
    exit 1
fi

if [ -z "${IMAGE_NAME}" ]; then
    echo "Error: --image-name is required"
    echo "Usage: ./build-base-image.sh --dockerfile <path> --image-name <name>"
    exit 1
fi

# Validate dockerfile exists
if [ ! -f "${DOCKERFILE}" ]; then
    echo "Error: Dockerfile not found: ${DOCKERFILE}"
    exit 1
fi

# Derived image values
FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${TAG_SUFFIX}"
CACHE_REPO="${REGISTRY}/${IMAGE_NAME}-cache"

# Build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
WEEK_NUMBER=$(date +%Y-W%V)

echo "=========================================="
echo "Building vLLM CI Base Image"
echo "=========================================="
echo "Image:            ${IMAGE_NAME}"
echo "Dockerfile:       ${DOCKERFILE}"
echo "CUDA:             ${CUDA_VERSION}"
echo "Python:           ${PYTHON_VERSION}"
echo "Ubuntu:           ${UBUNTU_VERSION}"
echo "Use Cache:        ${USE_CACHE}"
echo "Week:             ${WEEK_NUMBER}"
echo "Tag:              ${FULL_TAG}"
echo "=========================================="

# Build cache arguments
CACHE_ARGS=""
if [ "${USE_CACHE}" = true ]; then
    CACHE_ARGS="--cache-from type=registry,ref=${CACHE_REPO}:latest"
    CACHE_ARGS="${CACHE_ARGS} --cache-to type=registry,ref=${CACHE_REPO}:latest,mode=max,compression=zstd"
else
    echo "⚠️  Building without cache (weekly full rebuild)"
    # Still push to cache so next incremental build can use it
    CACHE_ARGS="--cache-to type=registry,ref=${CACHE_REPO}:latest,mode=max,compression=zstd"
fi

# Add FLASHINFER_VERSION if set (for runtime-base)
if [ -n "${FLASHINFER_VERSION}" ]; then
    EXTRA_BUILD_ARGS="${EXTRA_BUILD_ARGS} --build-arg FLASHINFER_VERSION=${FLASHINFER_VERSION}"
fi

# Build command
BUILD_CMD="docker buildx build \
    --file ${DOCKERFILE} \
    --platform linux/amd64 \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --build-arg GIT_SHA=${GIT_SHA} \
    ${EXTRA_BUILD_ARGS} \
    ${CACHE_ARGS} \
    --tag ${FULL_TAG} \
    --tag ${REGISTRY}/${IMAGE_NAME}:latest \
    --push \
    ."

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "Dry run - would execute:"
    echo "${BUILD_CMD}"
else
    echo ""
    echo "🔨 Starting build..."
    eval "${BUILD_CMD}"
    echo ""
    echo "=========================================="
    echo "✅ Build complete!"
    echo "   Image: ${FULL_TAG}"
    echo "=========================================="
fi
