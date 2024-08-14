# Building from Source

This document describes how to build the TensorRT-LLM backend and the Triton
TRT-LLM container from source. The Triton container includes TensorRT-LLM,
along with the TensorRT-LLM backend and the Python backend.

## Build the TensorRT-LLM Backend from source

Make sure TensorRT-LLM is installed before building the backend. Since the
version of TensorRT-LLM and the TensorRT-LLM backend has to be aligned, it is
recommended to directly use the Triton TRT-LLM container from NGC or build the
whole container from source as described below.

```bash
cd inflight_batcher_llm
bash scripts/build.sh
```

## Build the Docker Container

### Option 1. Build the NGC Triton TRT-LLM container

The below commands will build the same Triton TRT-LLM container as the one on the NGC.

You can update the arguments in the `build.sh` script to match the
versions you want to use.

```bash
cd tensorrtllm_backend
./build.sh
```

There should be a new image named `tritonserver` in your local Docker images.

#### Option 2. Build via Docker

The version of Triton Server used in this build option can be found in the
[Dockerfile](./dockerfile/Dockerfile.trt_llm_backend).

```bash
# Update the submodules
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive

# Use the Dockerfile to build the backend in a container
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
# For aarch64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm --build-arg TORCH_INSTALL_TYPE="src_non_cxx11_abi" -f dockerfile/Dockerfile.trt_llm_backend .
```
