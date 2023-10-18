<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# TensorRT-LLM Backend
The Triton backend for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
You can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend).
The goal of TensorRT-LLM Backend is to let you serve [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
models with Triton Inference Server. The [inflight_batcher_llm](./inflight_batcher_llm/)
directory contains the C++ implementation of the backend supporting inflight
batching, paged attention and more.

Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
[issues page](https://github.com/triton-inference-server/tensorrtllm_backend/issues).

## Building the TensorRT-LLM Backend

There are several ways to access the TensorRT-LLM Backend.

**Before Triton 23.10 release, please use [Option 3 to build TensorRT-LLM backend via CMake](#option-3-build-via-cmake)**

### Option 1. Run the Docker Container

**The NGC container will be available with Triton 23.10 release soon**

Starting with release 23.10, Triton includes a container with the TensorRT-LLM
Backend and Python Backend. This container should have everything to run a
TensorRT-LLM model. You can find this container on the
[Triton NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

### Option 2. Build via the build.py Script in Server Repo

**Building via the build.py script will be available with Triton 23.10 release soon**

You can follow steps described in the
[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
guide and use the
[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)
script.

A sample command to build a Triton Server container with all options enabled is
shown below, which will build the same TRT-LLM container as the one on the NGC.

```bash
BASE_CONTAINER_IMAGE_NAME=nvcr.io/nvidia/tritonserver:23.10-py3-min
TENSORRTLLM_BACKEND_REPO_TAG=r23.10
PYTHON_BACKEND_REPO_TAG=r23.10

# Run the build script. The flags for some features or endpoints can be removed if not needed.
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
              --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --image=base,${BASE_CONTAINER_IMAGE_NAME} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
```

The `BASE_CONTAINER_IMAGE_NAME` is the base image that will be used to build the
container. By default it is set to the most recent min image of Triton, on NGC,
that matches the Triton release you are building for. You can change it to a
different image if needed by setting the `--image` flag like the command below.
The `TENSORRTLLM_BACKEND_REPO_TAG` and `PYTHON_BACKEND_REPO_TAG` are the tags of
the TensorRT-LLM backend and Python backend repositories that will be used
to build the container. You can also remove the features or endpoints that you
don't need by removing the corresponding flags.

### Option 3. Build via CMake

```bash
# Update the submodules
cd tensorrtllm_backend
git submodule update --init --recursive
git lfs install
git lfs pull

# Patch the CMakeLists.txt file for different ABI builds
patch inflight_batcher_llm/CMakeLists.txt  < inflight_batcher_llm/CMakeLists.txt.patch

# Move the source code to the current directory
mv inflight_batcher_llm/src .
mv inflight_batcher_llm/cmake .
mv inflight_batcher_llm/CMakeLists.txt .

# Create a build directory and run cmake
mkdir build
cd build
cmake -DTRITON_BUILD=ON -DTRTLLM_BUILD_CONTAINER=nvcr.io/nvidia/tritonserver:23.09-py3-min -DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME> -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```

The resulting `install/backends/tensorrtllm directory` can be added to a
Triton installation as `/opt/tritonserver/backends/tensorrtllm` within the Triton
NGC container.

When building the TensorRT-LLM Backend with the flag `TRITON_BUILD` set to `ON`,
it will launch a separate docker image to build an appropriate TRT-LLM
implementation as part of the build. This setting is useful to avoid having
extra dependencies that are not needed for building the backend. The image used
to build the TRT-LLM is specified by the CMake variable
`TRTLLM_BUILD_CONTAINER`. It is recommended to use the Triton min image on the
NGC that matches the Triton release you are building for so that it contains
the required CUDA dependencies.

The following required Triton repositories will be pulled and used in
the build. If the CMake variables below are not specified, "main" branch
of those repositories will be used. `[tag]` should be the same
as the TensorRT-LLM backend repository branch that you are trying to compile.

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=[tag]`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=[tag]`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=[tag]`

## Using the TensorRT-LLM Backend

Below is an example of how to serve a TensorRT-LLM model with the Triton
TensorRT-LLM Backend on a 4-GPU environment. The example uses the GPT model from
the [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt).

### Prepare TensorRT-LLM engines

You can skip this step if you already have the engines ready.
Follow the [guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt) in
TensorRT-LLM repository for more details on how to to prepare the engines for deployment.

```bash
# Update the submodule TensorRT-LLM repository
git submodule update --init --recursive

# TensorRT-LLM is required for generating engines. You can skip this step if
# you already have the package installed. If you are generating engines within
# the Triton container, you have to install the TRT-LLM package.
pip install git+https://github.com/NVIDIA/TensorRT-LLM.git
mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

# Go to the tensorrt_llm/examples/gpt directory
cd tensorrt_llm/examples/gpt

# Download weights from HuggingFace Transformers
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# Convert weights from HF Tranformers to FT format
python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 4 --storage-type float16

# Build TensorRT engines
python3 build.py --model_dir=./c-model/gpt2/4-gpu/ \
                 --world_size=4 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir=engines/fp16/4-gpu
```

### Create the model repository

There are four models in the [`all_models/inflight_batcher_llm`](./all_models/inflight_batcher_llm/)
directory that will be used in this example:
- "preprocessing": This model is used for tokenizing, meaning the conversion from prompts(string) to input_ids(list of ints).
- "tensorrt_llm": This model is a wrapper of your TensorRT-LLM model and is used for inferencing
- "postprocessing": This model is used for de-tokenizing, meaning the conversion from output_ids(list of ints) to outputs(string).
- "ensemble": This model is used to chain the three models above together:
preprocessing -> tensorrt_llm -> postprocessing

To learn more about ensemble model, please see
[here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models).

```bash
# Create the model repository that will be used by the Triton server
cd tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/* triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/gpt/engines/fp16/4-gpu/* triton_model_repo/tensorrt_llm/1
```

### Modify the model configuration
The following table shows the fields that need to be modified before deployment:

*triton_model_repo/preprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |
| `tokenizer_type` | The type of the tokenizer for the model, `t5`, `auto` and `llama` are supported. In this example, the type should be set to `auto` |

*triton_model_repo/tensorrt_llm/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `decoupled` | Controls streaming. Decoupled mode must be set to `True` if using the streaming option from the client. |
| `gpt_model_type` | Set to `inflight_fused_batching` when enabling in-flight batching support. To disable in-flight batching, set to `V1` |
| `gpt_model_path` | Path to the TensorRT-LLM engines for deployment. In this example, the path should be set to `/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |

*triton_model_repo/postprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |
| `tokenizer_type` | The type of the tokenizer for the model, `t5`, `auto` and `llama` are supported. In this example, the type should be set to `auto` |

### Launch Triton server *within NGC container*

**The NGC container will be available with Triton 23.10 release soon**

Before the Triton 23.10 release, you can launch the Triton 23.09 container
`nvcr.io/nvidia/tritonserver:23.09-py3` and add the directory
`/opt/tritonserver/backends/tensorrtllm` within the container following the
instructions in [Option 3 Build via CMake](#option-3-build-via-cmake).

```bash
# Launch the Triton container
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-py3 bash

cd /tensorrtllm_backend
# --world_size is the number of GPUs you want to use for serving
python3 scripts/launch_triton_server.py --world_size=4 --model_repo=/tensorrtllm_backend/triton_model_repo
```

### Query the server with the Triton generate endpoint

**This feature will be available with Triton 23.10 release soon**

You can query the server using Triton's
[generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)
with a curl command based on the following general format within your client
environment/container:

```bash
curl -X POST localhost:8000/v2/models/${MODEL_NAME}/generate -d '{"{PARAM1_KEY}": "{PARAM1_VALUE}", ... }'
```

In the case of the models used in this example, you can replace MODEL_NAME with `ensemble`. Examining the
ensemble model's config.pbtxt file, you can see that 4 parameters are required to generate a response
for this model:

- "text_input": Input text to generate a response from
- "max_tokens": The number of requested output tokens
- "bad_words": A list of bad words (can be empty)
- "stop_words": A list of stop words (can be empty)

Therefore, we can query the server in the following way:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```

Which should return a result similar to (formatted for readability):
```json
{
  "model_name": "ensemble",
  "model_version": "1",
  "sequence_end": false,
  "sequence_id": 0,
  "sequence_start": false,
  "text_output": "What is machine learning?\n\nMachine learning is a method of learning by using machine learning algorithms to solve problems.\n\n"
}
```

### Utilize the provided client script to send a request

You can send requests to the "tensorrt_llm" model with the provided
[python client script](./inflight_batcher_llm/client/inflight_batcher_llm_client.py)
as following:

```bash
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer_dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
```

The result should be similar to the following:

```
Got completed request
output_ids =  [[28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257, 21221, 878, 3867, 284, 3576, 287, 262, 1903, 6303, 82, 13, 679, 468, 1201, 3111, 287, 10808, 287, 3576, 11, 6342, 11, 21574, 290, 968, 1971, 13, 198, 198, 1544, 318, 6405, 284, 262, 1966, 2746, 290, 14549, 11, 11735, 12, 44507, 11, 290, 468, 734, 1751, 11, 257, 4957, 11, 18966, 11, 290, 257, 3367, 11, 7806, 13, 198, 198, 50, 726, 263, 338, 3656, 11, 11735, 12, 44507, 11, 318, 257, 1966, 2746, 290, 14549, 13, 198, 198, 1544, 318, 11803, 416, 465, 3656, 11, 11735, 12, 44507, 11, 290, 511, 734, 1751, 11, 7806, 290, 18966, 13, 198, 198, 50, 726, 263, 373, 4642, 287, 6342, 11, 4881, 11, 284, 257, 4141, 2988, 290, 257, 2679, 2802, 13, 198, 198, 1544, 373, 15657, 379, 262, 23566, 38719, 293, 748, 1355, 14644, 12, 3163, 912, 287, 6342, 290, 262, 15423, 4189, 710, 287, 6342, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 50, 726, 263, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290]]
Input: Born in north-east France, Soyer trained as a
Output:  chef before moving to London in the early 1990s. He has since worked in restaurants in London, Paris, Milan and New York.

He is married to the former model and actress, Anna-Marie, and has two children, a daughter, Emma, and a son, Daniel.

Soyer's wife, Anna-Marie, is a former model and actress.

He is survived by his wife, Anna-Marie, and their two children, Daniel and Emma.

Soyer was born in Paris, France, to a French father and a German mother.

He was educated at the prestigious Ecole des Beaux-Arts in Paris and the Sorbonne in Paris.

He was a member of the French Academy of Sciences and the French Academy of Arts.

He was a member of the French Academy of Sciences and the French Academy of Arts.

Soyer was a member of the French Academy of Sciences and
```

You can also stop the generation process early by using the `--stop-after-ms` option to send a stop request after a few milliseconds:

```bash
python inflight_batcher_llm/client/inflight_batcher_llm_client.py --stop-after-ms 200 --request-output-len 200 --tokenizer_dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
```

You will find that the generation process is stopped early and therefore the number of generated tokens is lower than 200.
You can have a look at the client code to see how early stopping is achieved.

### Launch Triton server *within Slurm based clusters*

#### Prepare some scripts

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

srun --mpi=pmix --container-image nvcr.io/nvidia/tritonserver:23.10-trtllm-py3 \
    --container-mounts /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    --container-workdir /tensorrtllm_backend \
    --output logs/tensorrt_llm_%t.out \
    bash /tensorrtllm_backend/tensorrt_llm_triton.sh
```

`tensorrt_llm_triton.sh`
```bash
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="/tensorrtllm_backend/triton_model_repo"

${TRITONSERVER} --model-repository=${MODEL_REPO} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix${SLURM_PROCID}_
```

#### Submit a Slurm job

```bash
sbatch tensorrt_llm_triton.sub
```

When successfully deployed, the server produces logs similar to the following ones.
```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### Kill the Triton server

```bash
pgrep tritonserver | xargs kill -9
```

## Testing the TensorRT-LLM Backend
Please follow the guide in [`ci/README.md`](ci/README.md) to see how to run
the testing for TensorRT-LLM backend.
