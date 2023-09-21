# TensorRT-LLM Backend
The Triton backend for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

## Introduction

This document describes how to serve models by TensorRT-LLM Triton backend. This backend is only an interface to call TensorRT-LLM in Triton. The heavy lifting, in terms of implementation, can be found in the TensorRT-LLM source code.

## Setup Environment

### Prepare the repository

Clone the repository, and update submodules recursively.
```
git clone git@github.com:NVIDIA/TensorRT-LLM.git
git submodule update --init --recursive
```

### Build the Docker image.
```
cd tensorrtllm_backend
docker build -f dockerfile/Dockerfile.trt_llm_backend -t tritonserver:w_trt_llm_backend .
```

The rest of the documentation assumes that the Docker image has already been built.

### How to select the models
There are two models under `all_models/`:
- gpt: A Python implementation of the TensorRT-LLM Triton backend
- inflight_batcher_llm: A C++ implementation of the TensorRT-LLM Triton backend

### Prepare TensorRT-LLM engines
Follow the [guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) in TensorRT-LLM to prepare the engines for deployment.

For example, please find the details in the document of TensorRT-LLM GPT for instrutions to build GPT engines: [link](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt#usage)

### How to set the model configuration

**TensorRT-LLM Triton Serving Configuration: config.pbtxt**

- This will be loaded by Triton servers
- This mainly describes the server and TensorRT-LLM inference hyperparameters.

There are several components in each implemented backend, and there is a config.pbtxt for each component, take `all_models/inflight_batcher_llm` as an example:
- preprocessing: Used for tokenizing.
- tensorrt_llm: Inferencing.
- postprocessing: Used for de-tokenizing.
- ensemble: Connect preprocessing -> tensorrt_llm -> postprocessing

The following table shows the fields that need to be modified before deployment:

*all_models/inflight_batcher_llm/preprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model |
| `tokenizer_type` | The type of the tokenizer for the model, t5, auto and llama are supported |

*all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `decoupled` | Controls streaming. Decoupled mode must be set to true if using the streaming option from the client. |
| `gpt_model_type` | "inflight_fused_batching" or "V1" (disable in-flight batching) |
| `gpt_model_path` | Path to the TensorRT-LLM engines for deployment |

*all_models/inflight_batcher_llm/postprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model |
| `tokenizer_type` | The type of the tokenizer for the model, t5, auto and llama are supported |

## Run Serving on Single Node

### Launch the backend *within Docker*

```bash
# 1. Pull the docker image
nvidia-docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --shm-size=2g -v <your/path>:<mount/path> <image> bash

# 2. Modify parameters:
1. all_models/<model>/tensorrt_llm/config.pbtxt
2. all_models/<model>/preprocessing/config.pbtxt
3. all_models/<model>/postprocessing/config.pbtxt

# 3. Launch triton server
python3 scripts/launch_triton_server.py --world_size=<num_gpus> \
    --model_repo=all_models/<model>
```

### Launch the backend *within Slurm based clusters*
1. Prepare some scripts

`tensorrt_llm_triton.sub`
```bash
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J gpu-comparch-ftp:mgmn
#SBATCH -A gpu-comparch
#SBATCH -p luna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix --container-image <image> \
    --container-mounts <your/path>:<mount/path> \
    --container-workdir <workdir> \
    --output logs/tensorrt_llm_%t.out \
    bash <workdir>/tensorrt_llm_triton.sh
```

`tensorrt_llm_triton.sh`
```
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="<workdir>/triton_backend/"

${TRITONSERVER} --model-repository=${MODEL_REPO} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix${SLURM_PROCID}_
```

2. Submit a Slurm job
```
sbatch tensorrt_llm_triton.sub
```

When successfully deployed, the server produces logs similar to the following ones.
```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### Kill the backend

```bash
pgrep tritonserver | xargs kill -9
```

## C++ backend examples (support inflight batching)
Please follow the guide in [`inflight_batcher_llm/README.md`](inflight_batcher_llm/README.md).

## Python backend examples (not support inflight batching)

### GPT
```bash
cd tools/gpt/

rm -rf gpt2 && git clone https://huggingface.co/gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && \
    wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

python3 client.py \
    --text="Born in north-east France, Soyer trained as a" \
    --output_len=10 \
    --tokenizer_dir gpt2 \
    --tokenizer_type auto

# Exmaple output:
# [INFO] Latency: 92.278 ms
# Input: Born in north-east France, Soyer trained as a
# Output:  chef and a cook at the local restaurant, La
```
*Please note that the example outputs are only for reference, specific performance numbers depend on the GPU you're using.*

## Test

```bash
cd tools/gpt/

# Identity test
python3 identity_test.py \
    --batch_size=8 --start_len=128 --output_len=20
# Results:
# [INFO] Batch size: 8, Start len: 8, Output len: 10
# [INFO] Latency: 70.782 ms
# [INFO] Throughput: 113.023 sentences / sec

# Benchmark using Perf Analyzer
python3 gen_input_data.py
perf_analyzer -m tensorrt_llm \
    -b 8 --input-data input_data.json \
    --concurrency-range 1:10:2 \
    -u 'localhost:8000'

# Results:
# Concurrency: 1, throughput: 99.9875 infer/sec, latency 79797 usec
# Concurrency: 3, throughput: 197.308 infer/sec, latency 121342 usec
# Concurrency: 5, throughput: 259.077 infer/sec, latency 153693 usec
# Concurrency: 7, throughput: 286.18 infer/sec, latency 195011 usec
# Concurrency: 9, throughput: 307.067 infer/sec, latency 233354 usec
```
*Please note that the example outputs are only for reference, specific performance numbers depend on the GPU you're using.*
