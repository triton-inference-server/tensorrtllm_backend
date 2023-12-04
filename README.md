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

**Before Triton 23.10 release, please use [Option 3 to build TensorRT-LLM backend via Docker](#option-3-build-via-docker)**

### Option 1. Run the Docker Container

Starting with Triton 23.10 release, Triton includes a container with the TensorRT-LLM
Backend and Python Backend. This container should have everything to run a
TensorRT-LLM model. You can find this container on the
[Triton NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

### Option 2. Build via the build.py Script in Server Repo

Starting with Triton 23.10 release, you can follow steps described in the
[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
guide and use the
[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)
script.

A sample command to build a Triton Server container with all options enabled is
shown below, which will build the same TRT-LLM container as the one on the NGC.

```bash
BASE_CONTAINER_IMAGE_NAME=nvcr.io/nvidia/tritonserver:23.10-py3-min
TENSORRTLLM_BACKEND_REPO_TAG=release/0.5.0
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

### Option 3. Build via Docker

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
git lfs install
git lfs pull

# TensorRT-LLM is required for generating engines. You can skip this step if
# you already have the package installed. If you are generating engines within
# the Triton container, you have to install the TRT-LLM package.
(cd tensorrt_llm &&
    bash docker/common/install_cmake.sh &&
    export PATH=/usr/local/cmake/bin:$PATH &&
    python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt" &&
    pip3 install ./build/tensorrt_llm*.whl)

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

There are five models in the [`all_models/inflight_batcher_llm`](./all_models/inflight_batcher_llm/)
directory that will be used in this example:
- "preprocessing": This model is used for tokenizing, meaning the conversion from prompts(string) to input_ids(list of ints).
- "tensorrt_llm": This model is a wrapper of your TensorRT-LLM model and is used for inferencing
- "postprocessing": This model is used for de-tokenizing, meaning the conversion from output_ids(list of ints) to outputs(string).
- "ensemble": This model can be used to chain the preprocessing, tensorrt_llm and postprocessing models together.
- "tensorrt_llm_bls": This model can also be used to chain the preprocessing, tensorrt_llm and postprocessing models together. The BLS model has an optional parameter `accumulate_tokens` which can be used in streaming mode to call the preprocessing model with all accumulated tokens, instead of only one token. This might be necessary for certain tokenizers.

To learn more about ensemble and BLS models, please see the
[Ensemble Models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models) and [Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting) sections of the Triton Inference Server documentation.

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

### Launch Triton server

Please follow the option corresponding to the way you build the TensorRT-LLM backend.

#### Option 1. Launch Triton server *within Triton NGC container*

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash
```

#### Option 2. Launch Triton server *within the Triton container built via build.py script*

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend tritonserver bash
```

#### Option 3. Launch Triton server *within the Triton container built via Docker*

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend triton_trt_llm bash
```

Once inside the container, you can launch the Triton server with the following command:

```bash
cd /tensorrtllm_backend
# --world_size is the number of GPUs you want to use for serving
python3 scripts/launch_triton_server.py --world_size=4 --model_repo=/tensorrtllm_backend/triton_model_repo
```

When successfully deployed, the server produces logs similar to the following ones.
```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### Query the server with the Triton generate endpoint

Starting with Triton 23.10 release, you can query the server using Triton's
[generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)
with a curl command based on the following general format within your client
environment/container:

```bash
curl -X POST localhost:8000/v2/models/${MODEL_NAME}/generate -d '{"{PARAM1_KEY}": "{PARAM1_VALUE}", ... }'
```

In the case of the models used in this example, you can replace MODEL_NAME with `ensemble` or `tensorrt_llm_bls`. Examining the
`ensemble` and `tensorrt_llm_bls` model's config.pbtxt file, you can see that 4 parameters are required to generate a response
for this model:

- "text_input": Input text to generate a response from
- "max_tokens": The number of requested output tokens
- "bad_words": A list of bad words (can be empty)
- "stop_words": A list of stop words (can be empty)

Therefore, we can query the server in the following way:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```
if using the `ensemble` model or
```
curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```
if using the `tensorrt_llm_bls` model.

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
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
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
python inflight_batcher_llm/client/inflight_batcher_llm_client.py --stop-after-ms 200 --request-output-len 200 --tokenizer-dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
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
#SBATCH -J <REPLACE WITH YOUR JOB's NAME>
#SBATCH -A <REPLACE WITH YOUR ACCOUNT's NAME>
#SBATCH -p <REPLACE WITH YOUR PARTITION's NAME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix \
    --container-image triton_trt_llm \
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

You might have to contact your cluster's administrator to help you customize the above script.

### Kill the Triton server

```bash
pkill tritonserver
```

## Triton Metrics
Starting with the 23.11 release of Triton, users can now obtain TRT LLM Batch Manager [statistics](https://github.com/NVIDIA/TensorRT-LLM/blob/ffd5af342a817a2689d38e4af2cc59ded877e339/docs/source/batch_manager.md#statistics) by querying the Triton metrics endpoint. This can be accomplished by launching a Triton server in any of the ways described above (ensuring the build code / container is 23.11 or later) and querying the sever with the generate endpoint. Upon receiving a successful response, you can query the metrics endpoint by entering the following:
```bash
curl localhost:8002/metrics
```
Batch manager statistics are reported by the metrics endpoint in fields that are prefixed with `nv_trt_llm_`. Your output for these fields should look similar to the following (assuming your model is an inflight batcher model):
```bash
# HELP nv_trt_llm_request_statistics TRT LLM request metrics
# TYPE nv_trt_llm_request_statistics gauge
nv_trt_llm_request_statistics{model="tensorrt_llm",request_type="context",version="1"} 1
nv_trt_llm_request_statistics{model="tensorrt_llm",request_type="scheduled",version="1"} 1
nv_trt_llm_request_statistics{model="tensorrt_llm",request_type="max",version="1"} 512
nv_trt_llm_request_statistics{model="tensorrt_llm",request_type="active",version="1"} 0
# HELP nv_trt_llm_runtime_memory_statistics TRT LLM runtime memory metrics
# TYPE nv_trt_llm_runtime_memory_statistics gauge
nv_trt_llm_runtime_memory_statistics{memory_type="pinned",model="tensorrt_llm",version="1"} 0
nv_trt_llm_runtime_memory_statistics{memory_type="gpu",model="tensorrt_llm",version="1"} 1610236
nv_trt_llm_runtime_memory_statistics{memory_type="cpu",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_kv_cache_block_statistics TRT LLM KV cache block metrics
# TYPE nv_trt_llm_kv_cache_block_statistics gauge
nv_trt_llm_kv_cache_block_statistics{kv_cache_block_type="tokens_per",model="tensorrt_llm",version="1"} 64
nv_trt_llm_kv_cache_block_statistics{kv_cache_block_type="used",model="tensorrt_llm",version="1"} 1
nv_trt_llm_kv_cache_block_statistics{kv_cache_block_type="free",model="tensorrt_llm",version="1"} 6239
nv_trt_llm_kv_cache_block_statistics{kv_cache_block_type="max",model="tensorrt_llm",version="1"} 6239
# HELP nv_trt_llm_inflight_batcher_statistics TRT LLM inflight_batcher-specific metrics
# TYPE nv_trt_llm_inflight_batcher_statistics gauge
nv_trt_llm_inflight_batcher_statistics{inflight_batcher_specific_metric="micro_batch_id",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_statistics{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_statistics{inflight_batcher_specific_metric="total_context_tokens",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_general_statistics General TRT LLM statistics
# TYPE nv_trt_llm_general_statistics gauge
nv_trt_llm_general_statistics{general_type="iteration_counter",model="tensorrt_llm",version="1"} 0
nv_trt_llm_general_statistics{general_type="timestamp",model="tensorrt_llm",version="1"} 1700074049
```
If, instead, you launched a V1 model, your output will look similar to the output above except the inflight batcher related fields will be replaced with something similar to the following:
```bash
# HELP nv_trt_llm_v1_statistics TRT LLM v1-specific metrics
# TYPE nv_trt_llm_v1_statistics gauge
nv_trt_llm_v1_statistics{model="tensorrt_llm",v1_specific_metric="total_generation_tokens",version="1"} 20
nv_trt_llm_v1_statistics{model="tensorrt_llm",v1_specific_metric="empty_generation_slots",version="1"} 0
nv_trt_llm_v1_statistics{model="tensorrt_llm",v1_specific_metric="total_context_tokens",version="1"} 5
```
Please note that as of the 23.11 Triton release, a link between base Triton metrics (such as inference request count and latency) is being actively developed, but is not yet supported.
As such, the following fields will report 0:
```bash
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_count Number of inferences performed (does not include cached requests)
# TYPE nv_inference_count counter
nv_inference_count{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_request_duration_us Cumulative inference request duration in microseconds (includes cached requests)
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_queue_duration_us Cumulative inference queuing duration in microseconds (includes cached requests)
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_input_duration_us Cumulative compute input duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_input_duration_us counter
nv_inference_compute_input_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_infer_duration_us Cumulative compute inference duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_infer_duration_us counter
nv_inference_compute_infer_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_output_duration_us Cumulative inference compute output duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_output_duration_us counter
nv_inference_compute_output_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_pending_request_count Instantaneous number of pending requests awaiting execution per-model.
# TYPE nv_inference_pending_request_count gauge
nv_inference_pending_request_count{model="tensorrt_llm",version="1"} 0
```

## Testing the TensorRT-LLM Backend
Please follow the guide in [`ci/README.md`](ci/README.md) to see how to run
the testing for TensorRT-LLM backend.

## Known Issues

  * Recent API changes may currently lead to excessive warnings in the logs of the Triton TRT-LLM backend. On each inference request, TRT-LLM will issue the warning "Invalid tensor name in InferenceRequest: input_lengths". As a temporary fix, these warnings can be disabled by setting "TLLM_LOG_LEVEL=ERROR" in the environment.
