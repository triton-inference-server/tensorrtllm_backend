#!/bin/bash

git lfs install
git submodule update --init --recursive

# Default values will be used if not set
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/tritonserver:24.07-py3-min}
PYTORCH_IMAGE=${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:24.07-py3}
TRT_VERSION=${TRT_VERSION:-10.4.0.26}
TRT_URL_x86=${TRT_URL_x86:-https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-12.6.tar.gz}
TRT_URL_ARM=${TRT_URL_ARM:-https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-${TRT_VERSION}.ubuntu-24.04.aarch64-gnu.cuda-12.6.tar.gz}

# Build the TRT-LLM base image that has TRT-LLM installed and will be used as
# the base image for building Triton server and TRT-LLM backend.
docker build -t trtllm_base \
             --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             --build-arg PYTORCH_IMAGE="${PYTORCH_IMAGE}" \
             --build-arg TRT_VER="${TRT_VERSION}" \
             --build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
             --build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
             -f dockerfile/Dockerfile.triton.trt_llm_backend .

# Clone the Triton server repository on the same level as the TRT-LLM backend repository.
cd ../
# Need to use the aligned version of the Triton server repository.
# Refer to the support matrix for the aligned version: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
TRITON_SERVER_REPO_TAG=${TRITON_SERVER_REPO_TAG:-r24.09}
git clone -b ${TRITON_SERVER_REPO_TAG} https://github.com/triton-inference-server/server.git
cd server

# The `TRTLLM_BASE_IMAGE` is the base image that will be used to build the
# container. The `TENSORRTLLM_BACKEND_REPO_TAG` and `PYTHON_BACKEND_REPO_TAG` are
# the tags of the TensorRT-LLM backend and Python backend repositories that will
# be used to build the container.
TRTLLM_BASE_IMAGE=${TRTLLM_BASE_IMAGE:-trtllm_base}
TENSORRTLLM_BACKEND_REPO_TAG=${TENSORRTLLM_BACKEND_REPO_TAG:-v0.13.0}
PYTHON_BACKEND_REPO_TAG=${PYTHON_BACKEND_REPO_TAG:-r24.09}

# The flags for some features or endpoints can be removed if not needed.
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
              --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
              --backend=ensemble --enable-gpu --no-container-pull \
              --repoagent=checksum --cache=local --cache=redis \
              --image=base,${TRTLLM_BASE_IMAGE} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
