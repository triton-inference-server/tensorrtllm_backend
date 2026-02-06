# Building from Source

This document describes how to build the TensorRT-LLM backend and the Triton
TRT-LLM container from source. The Triton container includes TensorRT-LLM,
along with the TensorRT-LLM backend and the Python backend.

## Build the TensorRT-LLM Backend from source

Make sure TensorRT-LLM is installed before building the backend. Since the
version of TensorRT-LLM and the TensorRT-LLM backend has to be aligned, it is
recommended to directly use the Triton TRT-LLM container from NGC or build the
whole container from source as described below in the Build the Docker Container
section.

```bash
cd tensorrt_llm/triton_backend/inflight_batcher_llm
bash scripts/build.sh
```

## Build the Docker Container

#### Build via Docker

You can build the container using the instructions in the [TensorRT-LLM Docker Build](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/README.md)
with `tritonrelease` stage. Please make sure to add CUDA_ARCHS flag for your GPU, for example if compute capability of your GPU is 89:

```bash
cd tensorrt_llm/
make -C docker tritonrelease_build CUDA_ARCHS='89-real'
```
