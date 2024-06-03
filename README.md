<!--
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Accessing the TensorRT-LLM Backend

There are several ways to access the TensorRT-LLM Backend.

**Before Triton 23.10 release, please use [Option 3 to build TensorRT-LLM backend via Docker](#option-3-build-via-docker).**

### Run the Pre-built Docker Container

Starting with Triton 23.10 release, Triton includes a container with the TensorRT-LLM
Backend and Python Backend. This container should have everything to run a
TensorRT-LLM model. You can find this container on the
[Triton NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

### Build the Docker Container

#### Option 1. Build via the `build.py` Script in Server Repo

Starting with Triton 23.10 release, you can follow steps described in the
[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
guide and use the
[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)
script to build the TRT-LLM backend.

The below commands will build the same Triton TRT-LLM container as the one on the NGC.

```bash
# Prepare the TRT-LLM base image using the dockerfile from tensorrtllm_backend.
cd tensorrtllm_backend
# Specify the build args for the dockerfile.
BASE_IMAGE=nvcr.io/nvidia/pytorch:24.03-py3
TRT_VERSION=10.0.1.6
TRT_URL_x86=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
TRT_URL_ARM=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.ubuntu-22.04.aarch64-gnu.cuda-12.4.tar.gz

docker build -t trtllm_base \
             --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             --build-arg TRT_VER="${TRT_VERSION}" \
             --build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
             --build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
             -f dockerfile/Dockerfile.triton.trt_llm_backend .

# Run the build script from Triton Server repo. The flags for some features or
# endpoints can be removed if not needed. Please refer to the support matrix to
# see the aligned versions: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
TRTLLM_BASE_IMAGE=trtllm_base
TENSORRTLLM_BACKEND_REPO_TAG=rel
PYTHON_BACKEND_REPO_TAG=r24.04

cd server
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
              --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --no-container-pull \
              --image=base,${TRTLLM_BASE_IMAGE} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
```

The `TRTLLM_BASE_IMAGE` is the base image that will be used to build the
container. The `TENSORRTLLM_BACKEND_REPO_TAG` and `PYTHON_BACKEND_REPO_TAG` are
the tags of the TensorRT-LLM backend and Python backend repositories that will
be used to build the container. You can also remove the features or endpoints
that you don't need by removing the corresponding flags.

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

# Convert weights from HF Tranformers to TensorRT-LLM checkpoint
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 4 \
        --output_dir ./c-model/gpt2/fp16/4-gpu

# Build TensorRT engines
trtllm-build --checkpoint_dir ./c-model/gpt2/fp16/4-gpu \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --output_dir engines/fp16/4-gpu
```

### Create the model repository

There are five models in the [`all_models/inflight_batcher_llm`](./all_models/inflight_batcher_llm/)
directory that will be used in this example:

#### preprocessing

This model is used for tokenizing, meaning the conversion from
prompts(string) to input_ids(list of ints).

#### tensorrt_llm

This model is a wrapper of your TensorRT-LLM model and is used
for inferencing.
Input specification can be found [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/advanced/inference-request.md)

#### postprocessing

This model is used for de-tokenizing, meaning the conversion
from output_ids(list of ints) to outputs(string).

#### ensemble

This model can be used to chain the preprocessing, tensorrt_llm
and postprocessing models together.

#### tensorrt_llm_bls

This model can also be used to chain the preprocessing,
tensorrt_llm and postprocessing models together.

When using the BLS model instead of the ensemble, you should set the number of model instances to
the maximum batch size supported by the TRT engine to allow concurrent request execution. This
can be done by modifying the `count` value in the `instance_group` section of the BLS model `config.pbtxt`.

The BLS model has an optional parameter `accumulate_tokens` which can be used in streaming mode to call the
postprocessing model with all accumulated tokens, instead of only one token.
This might be necessary for certain tokenizers.

The BLS model supports speculative decoding.  Target and draft triton models are set with the parameters `tensorrt_llm_model_name` `tensorrt_llm_draft_model_name`.  Speculative decoding is performed by setting `num_draft_tokens` in the request.  `use_draft_logits` may be set to use logits comparison speculative decoding. Note that `return_generation_logits` and `return_context_logits` are not supported when using speculative decoding.

BLS Inputs

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_input` | [ -1 ] | `string` | Prompt text |
| `max_tokens` | [ -1 ] | `int32` | number of tokens to generate |
| `bad_words` | [2, num_bad_words] | `int32` | Bad words list |
| `stop_words` | [2, num_stop_words] | `int32` | Stop words list |
| `end_id` | [1] | `int32` | End token Id. If not specified, defaults to -1 |
| `pad_id` | [1] | `int32` | Pad token Id |
| `temperature` | [1] | `float32` | Sampling Config param: `temperature` |
| `top_k` | [1] | `int32` | Sampling Config param: `topK` |
| `top_p` | [1] | `float32` | Sampling Config param: `topP` |
| `len_penalty` | [1] | `float32` | Sampling Config param: `lengthPenalty` |
| `repetition_penalty` | [1] | `float` | Sampling Config param: `repetitionPenalty` |
| `min_length` | [1] | `int32_t` | Sampling Config param: `minLength` |
| `presence_penalty` | [1] | `float` | Sampling Config param: `presencePenalty` |
| `frequency_penalty` | [1] | `float` | Sampling Config param: `frequencyPenalty` |
| `random_seed` | [1] | `uint64_t` | Sampling Config param: `randomSeed` |
| `return_log_probs` | [1] | `bool` | When `true`, include log probs in the output |
| `return_context_logits` | [1] | `bool` | When `true`, include context logits in the output |
| `return_generation_logits` | [1] | `bool` | When `true`, include generation logits in the output |
| `beam_width` | [1] | `int32_t` | (Default=1) Beam width for this request; set to 1 for greedy sampling |
| `stream` | [1] | `bool` | (Default=`false`). When `true`, stream out tokens as they are generated. When `false` return only when the full generation has completed.  |
| `prompt_embedding_table` | [1] | `float16` (model data type) | P-tuning prompt embedding table |
| `prompt_vocab_size` | [1] | `int32` | P-tuning prompt vocab size |
| `lora_task_id` | [1] | `uint64` | Task ID for the given lora_weights.  This ID is expected to be globally unique.  To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.  The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`. If the cache is full the oldest LoRA will be evicted to make space for new ones.  An error is returned if `lora_task_id` is not cached |
| `lora_weights` | [ num_lora_modules_layers, D x Hi + Ho x D ] | `float` (model data type) | weights for a lora adapter. see [lora docs](lora.md#lora-tensor-format-details) for more details. |
| `lora_config` | [ num_lora_modules_layers, 3] | `int32t` | lora configuration tensor. `[ module_id, layer_idx, adapter_size (D aka R value) ]` see [lora docs](lora.md#lora-tensor-format-details) for more details. |
| `embedding_bias_words` | [-1] | `string` | Embedding bias words |
| `embedding_bias_weights` | [-1] | `float32` | Embedding bias weights |
| `num_draft_tokens` | [1] | int32 | number of tokens to get from draft model during speculative decoding |
| `use_draft_logits` | [1] | `bool` | use logit comparison during speculative decoding |

BLS Outputs

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_output` | [-1] | `string` | text output |
| `cum_log_probs` | [-1] | `float` | cumulative probabilities for each output |
| `output_log_probs` | [beam_width, -1] | `float` | log probabilities for each output |
| `context_logits` | [-1, vocab_size] | `float` |  context logits for input |
| `generation_logtis` | [beam_width, seq_len, vocab_size] | `float` | generatiion logits for each output |

To learn more about ensemble and BLS models, please see the
[Ensemble Models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
and [Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
sections of the Triton Inference Server documentation.

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
The following table shows the fields that may to be modified before deployment:

*triton_model_repo/preprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |

*triton_model_repo/tensorrt_llm/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `gpt_model_type` | Mandatory. Set to `inflight_fused_batching` when enabling in-flight batching support. To disable in-flight batching, set to `V1` |
| `gpt_model_path` | Mandatory. Path to the TensorRT-LLM engines for deployment. In this example, the path should be set to `/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |
| `batch_scheduler_policy` | Mandatory. Set to `max_utilization` to greedily pack as many requests as possible in each current in-flight batching iteration. This maximizes the throughput but may result in overheads due to request pause/resume if KV cache limits are reached during execution. Set to `guaranteed_no_evict` to guarantee that a started request is never paused.|
| `decoupled` | Optional (default=`false`). Controls streaming. Decoupled mode must be set to `true` if using the streaming option from the client. |
| `max_beam_width` | Optional (default=1). The maximum beam width that any request may ask for when using beam search.|
| `max_tokens_in_paged_kv_cache` | Optional (default=unspecified). The maximum size of the KV cache in number of tokens. If unspecified, value is interpreted as 'infinite'. KV cache allocation is the min of max_tokens_in_paged_kv_cache and value derived from kv_cache_free_gpu_mem_fraction below. |
| `max_attention_window_size` | Optional (default=max_sequence_length). When using techniques like sliding window attention, the maximum number of tokens that are attended to generate one token. Defaults attends to all tokens in sequence. |
| `kv_cache_free_gpu_mem_fraction` | Optional (default=0.9). Set to a number between 0 and 1 to indicate the maximum fraction of GPU memory (after loading the model) that may be used for KV cache.|
| `exclude_input_in_output` | Optional (default=`false`). Set to `true` to only return completion tokens in a response. Set to `false` to return the prompt tokens concatenated with the generated tokens  |
| `cancellation_check_period_ms` | Optional (default=100). The time for cancellation check thread to sleep before doing the next check. It checks if any of the current active requests are cancelled through triton and prevent further execution of them. |
| `stats_check_period_ms` | Optional (default=100). The time for the statistics reporting thread to sleep before doing the next check. |
| `iter_stats_max_iterations` | Optional (default=executor::kDefaultIterStatsMaxIterations). The numbers of iteration stats to be kept. |
| `request_stats_max_iterations` | Optional (default=executor::kDefaultRequestStatsMaxIterations). The numbers of request stats to be kept. |
| `normalize_log_probs` | Optional (default=`true`). Set to `false` to skip normalization of `output_log_probs`  |
| `enable_chunked_context` | Optional (default=`false`). Set to `true` to enable context chunking. |
| `gpu_device_ids` | Optional (default=unspecified). Comma-separated list of GPU IDs to use for this model. If not provided, the model will use all visible GPUs. |
| `decoding_mode` | Optional. Set to one of the following: `{top_k, top_p, top_k_top_p, beam_search, medusa}` to select the decoding mode. The `top_k` mode exclusively uses Top-K algorithm for sampling, The `top_p` mode uses exclusively Top-P algorithm for sampling. The top_k_top_p mode employs both Top-K and Top-P algorithms, depending on the runtime sampling params of the request. Note that the `top_k_top_p option` requires more memory and has a longer runtime than using `top_k` or `top_p` individually; therefore, it should be used only when necessary. `beam_search` uses beam search algorithm. If not specified, the default is to use `top_k_top_p` if `max_beam_width == 1`; otherwise, `beam_search` is used. When Medusa model is used, `medusa` decoding mode should be set. However, TensorRT-LLM detects loaded Medusa model and overwrites decoding mode to `medusa` with warning. |
| `medusa_choices` | Optional. To specify Medusa choices tree in the format of e.g. "{0, 0, 0}, {0, 1}". By default, mc_sim_7b_63 choices are used. |
| `lora_cache_optimal_adapter_size` | Optional (default=8) Optimal adapter size used to size cache pages. Typically optimally sized adapters will fix exactly into 1 cache page. |
| `lora_cache_max_adapter_size` | Optional (default=64) Used to set the minimum size of a cache page.  Pages must be at least large enough to fit a single module, single later adapter_size `maxAdapterSize` row of weights. |
| `lora_cache_gpu_memory_fraction` | Optional (default=0.05) Fraction of GPU memory used for LoRA cache. Computed as a fraction of left over memory after engine load, and after KV cache is loaded |
| `lora_cache_host_memory_bytes` | Optional (default=1G) Size of host LoRA cache in bytes |
| `gpu_weights_percent` | Optional (default=1.0). Set to a number between 0.0 and 1.0 to specify the percentage of weights that reside on GPU instead of CPU and streaming load during runtime. Values less than 1.0 are only supported for an engine built with `weight_streaming` on. |

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

In order to use multiple TensorRT-LLM models, use the `--multi-model` option. The `--world_size` must be 1 as the TensorRT-LLM backend will dynamically launch TensorRT-LLM workers as needed.

```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --model_repo=/tensorrtllm_backend/triton_model_repo --multi-model
```

When using the `--multi-model` option, the Triton model repository can contain multiple TensorRT-LLM models. When running multiple TensorRT-LLM models, the `gpu_device_ids` parameter should be specified in the models `config.pbtxt` configuration files. It is up to you to ensure there is no overlap between allocated GPU IDs.

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

#### Early stopping
You can also stop the generation process early by using the `--stop-after-ms`
option to send a stop request after a few milliseconds:

```bash
python inflight_batcher_llm/client/inflight_batcher_llm_client.py --stop-after-ms 200 --request-output-len 200 --tokenizer-dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
```

You will find that the generation process is stopped early and therefore the
number of generated tokens is lower than 200. You can have a look at the
client code to see how early stopping is achieved.

#### Return context logits and/or generation logits
If you want to get context logits and/or generation logits, you need to enable `--gather_context_logits` and/or `--gather_generation_logits` when building the engine (or `--gather_all_token_logits` to enable both at the same time). For more setting details about these two flags, please refer to [build.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/commands/build.py) or [gpt_runtime](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/gpt_runtime.md).

After launching the server, you could get the output of logits by passing the corresponding parameters `--return-context-logits` and/or `--return-generation-logits` in the client scripts ([end_to_end_grpc_client.py](./inflight_batcher_llm/client/end_to_end_grpc_client.py) and [inflight_batcher_llm_client.py](./inflight_batcher_llm/client/inflight_batcher_llm_client.py)). For example:
```bash
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 20 --tokenizer-dir /path/to/tokenizer/ \
--return-context-logits \
--return-generation-logits
```

The result should be similar to the following:
```
Input sequence:  [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257]
Got completed request
Input: Born in north-east France, Soyer trained as a
Output beam 0:  has since worked in restaurants in London,
Output sequence:  [21221, 878, 3867, 284, 3576, 287, 262, 1903, 6303, 82, 13, 679, 468, 1201, 3111, 287, 10808, 287, 3576, 11]
context_logits.shape: (1, 12, 50257)
context_logits: [[[ -65.9822     -62.267445   -70.08991   ...  -76.16964    -78.8893
    -65.90678  ]
  [-103.40278   -102.55243   -106.119026  ... -108.925415  -109.408585
   -101.37687  ]
  [ -63.971176   -64.03466    -67.58809   ...  -72.141235   -71.16892
    -64.23846  ]
  ...
  [ -80.776375   -79.1815     -85.50916   ...  -87.07368    -88.02817
    -79.28435  ]
  [ -10.551408    -7.786484   -14.524468  ...  -13.805856   -15.767286
     -7.9322424]
  [-106.33096   -105.58956   -111.44852   ... -111.04858   -111.994194
   -105.40376  ]]]
generation_logits.shape: (1, 1, 20, 50257)
generation_logits: [[[[-106.33096  -105.58956  -111.44852  ... -111.04858  -111.994194
    -105.40376 ]
   [ -77.867424  -76.96638   -83.119095 ...  -87.82542   -88.53957
     -75.64877 ]
   [-136.92282  -135.02484  -140.96051  ... -141.78284  -141.55045
    -136.01668 ]
   ...
   [-100.03721   -98.98237  -105.25507  ... -108.49254  -109.45882
     -98.95136 ]
   [-136.78777  -136.16165  -139.13437  ... -142.21495  -143.57468
    -134.94667 ]
   [  19.222942   19.127287   14.804495 ...   10.556551    9.685863
      19.625107]]]]
```


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
Starting with the 23.11 release of Triton, users can now obtain TRT LLM Batch
Manager [statistics](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/advanced/batch-manager.md#statistics)
by querying the Triton metrics endpoint. This can be accomplished by launching
a Triton server in any of the ways described above (ensuring the build code /
container is 23.11 or later) and querying the server. Upon receiving a
successful response, you can query the metrics endpoint by entering the
following:
```bash
curl localhost:8002/metrics
```
Batch manager statistics are reported by the metrics endpoint in fields that
are prefixed with `nv_trt_llm_`. Your output for these fields should look
similar to the following (assuming your model is an inflight batcher model):
```bash
# HELP nv_trt_llm_request_metrics TRT LLM request metrics
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="context",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="scheduled",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="max",version="1"} 512
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="active",version="1"} 0
# HELP nv_trt_llm_runtime_memory_metrics TRT LLM runtime memory metrics
# TYPE nv_trt_llm_runtime_memory_metrics gauge
nv_trt_llm_runtime_memory_metrics{memory_type="pinned",model="tensorrt_llm",version="1"} 0
nv_trt_llm_runtime_memory_metrics{memory_type="gpu",model="tensorrt_llm",version="1"} 1610236
nv_trt_llm_runtime_memory_metrics{memory_type="cpu",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_kv_cache_block_metrics TRT LLM KV cache block metrics
# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="tokens_per",model="tensorrt_llm",version="1"} 64
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used",model="tensorrt_llm",version="1"} 1
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="free",model="tensorrt_llm",version="1"} 6239
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="max",model="tensorrt_llm",version="1"} 6239
# HELP nv_trt_llm_inflight_batcher_metrics TRT LLM inflight_batcher-specific metrics
# TYPE nv_trt_llm_inflight_batcher_metrics gauge
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="micro_batch_id",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="total_context_tokens",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_general_metrics General TRT LLM metrics
# TYPE nv_trt_llm_general_metrics gauge
nv_trt_llm_general_metrics{general_type="iteration_counter",model="tensorrt_llm",version="1"} 0
nv_trt_llm_general_metrics{general_type="timestamp",model="tensorrt_llm",version="1"} 1700074049
```
If, instead, you launched a V1 model, your output will look similar to the
output above except the inflight batcher related fields will be replaced
with something similar to the following:
```bash
# HELP nv_trt_llm_v1_metrics TRT LLM v1-specific metrics
# TYPE nv_trt_llm_v1_metrics gauge
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="total_generation_tokens",version="1"} 20
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="empty_generation_slots",version="1"} 0
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="total_context_tokens",version="1"} 5
```
Please note that versions of Triton prior to the 23.12 release do not
support base Triton metrics. As such, the following fields will report 0:
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
