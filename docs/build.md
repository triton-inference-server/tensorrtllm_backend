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

> [!CAUTION]
> [build.sh](../build.sh) is currently not working and will be fixed in the next weekly update.

#### Build via Docker

You can build the container using the instructions in the [TensorRT-LLM Docker Build](../tensorrt_llm/docker/README.md)
with `tritonrelease` stage.

```bash
cd tensorrt_llm/
make -C docker tritonrelease_build
```
