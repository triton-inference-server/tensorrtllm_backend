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
!-->

# Running Multiple Instances of the LLaMa Model

This document describes how you can run multiple instances of
LLaMa model on single and multiple GPUs running on the
same machine. The guide focuses on the following scenarios:

* [Running multiple instances of LLaMa model on a single GPU](#running-multiple-instances-of-llama-model-on-a-single-gpu).
* [Running multiple instances of LLaMa model on multiple GPUs](#running-multiple-instances-of-llama-model-on-multiple-gpus):

  a. Using [Orchestrator mode](#orchestrator-mode).

  b. Using [Leader mode](#leader-mode).

## Running multiple instances of LLaMa model on a single GPU

1. Setup the model repository as described in [LLaMa Guide](./llama.md).

2. Increase the number of instances for the `instance_group` parameter for
the `tensorrt_llm` model.

3. Start the triton server:

```bash
# Replace the <gpu> with the gpu you want to use for this model.
CUDA_VISIBLE_DEVICES=<gpu> tritonserver --model-repository `pwd`/llama_ifb &
```

This would create multiple instances of the `tensorrt_llm` model, running on the
same GPU.

> **Note**
>
> Running multiple instances of a single model is generally not
> recommended. If you choose to do this, you need to ensure the GPU has enough
> resources for multiple copies of a single model. The performance implications
> of running multiple models on the same GPU are unpredictable.

> **Note**
>
> For production deployments please make sure to adjust the
> `max_tokens_in_paged_kv_cache` parameter, otherwise you may run out of GPU
> memory since TensorRT-LLM by default may use 90% of GPU for KV-Cache for each
> model instance. Additionally, if you rely on `kv_cache_free_gpu_mem_fraction`
> the memory allocated to each instance will depend on the order in which instances are loaded.

4. Run the test client to measure performance:

```bash
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json --max-input-len 500
```

If you plan to use the BLS version instead of the ensemble model, you might also
need to adjust the number of model instances for the `tensorrt_llm_bls` model.
The default value only allows a single request for the whole pipeline which
might increase the latency and reduce the throughput.

5. Kill the server:

```bash
pgrep tritonserver | xargs kill
```

## Running multiple instances of LLaMa model on multiple GPUs

Unlike other Triton backend models, the TensorRT-LLM backend does not support
using `instance_group` setting for determining the placement of model instances
on different GPUs. In this section, we demonstrate how you can use
[Leader Mode](../README.md#leader-mode) and [Orchestrator Mode](../README.md#orchestrator-mode)
for running multiple instances of a LLaMa model on different GPUs.

For this section, let's assume that we have four GPUs and the CUDA device ids
are 0, 1, 2, and 3.  We will be launching two instances of the LLaMa2-7b model
with tensor parallelism equal to 2. The first instance will run on GPUs 0 and 1
and the second instance will run on GPUs 2 and 3.

1. Create the engines:

```bash
# Update if the model is not available in huggingface cache
export HF_LLAMA_MODEL=`python3 -c "from pathlib import Path; from huggingface_hub import hf_hub_download; print(Path(hf_hub_download('meta-llama/Llama-2-7b-hf', filename='config.json')).parent)"`

export UNIFIED_CKPT_PATH=/tmp/ckpt/llama/7b-2tp-2gpu/
export ENGINE_PATH=/tmp/engines/llama/7b-2tp-2gpu/

# Create the checkpoint
python tensorrt_llm/examples/models/core/llama/convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16 \
                             --tp_size 2

# Build the engines
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --kv_cache_type paged \
             --max_batch_size 64
```

2. Setup the model repository:

```bash
# Setup the model repository for the first instance.
cp tensorrt_llm/triton_backend/ci/all_models/inflight_batcher_llm/ llama_ifb -r

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
```

### Leader Mode

For leader mode, we will launch two separate `mpirun` commands to launch two
separate Triton servers, one for each GPU (4 Triton Server instances in total).
We also need to use a reverse proxy in front of them to load balance the requests
between the servers.

3a. Launch the servers:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 2 --model_repo=llama_ifb/ --http_port 8000 --grpc_port 8001 --metrics_port 8004
CUDA_VISIBLE_DEVICES=2,3 python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 2 --model_repo=llama_ifb/ --http_port 8002 --grpc_port 8003 --metrics_port 8005
```

4a. Install NGINX:

```bash
apt update
apt install nginx -y
```

5a. Setup the NGINX configuration and store it in `/etc/nginx/sites-available/tritonserver`:

```conf
upstream tritonserver {
    server localhost:8000;
    server localhost:8002;
}

server {
    listen 8080;

    location / {
        proxy_pass http://tritonserver;
    }
}
```

6a. Create a symlink and restart NGINX to enable the configuration:

```
ln -s /etc/nginx/sites-available/tritonserver /etc/nginx/sites-enabled/tritonserver
service nginx restart
```

7a. Run the test client to measure performance:

```bash
pip3 install tritonclient[all]

# Test the load on all the servers
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8080

# Test the load on one of the servers
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8000
```

8a. Kill the server:

```bash
pgrep mpirun | xargs kill
```

### Orchestrator Mode

With orchestrator mode, there are two options for running multiple instances
of a single model:

1. Creating separate Triton models

2. Starting from the 24.08 release, you can use Triton `instance_group` field to specify the number TRT-LLM model instances. With that option, the load balancing decision will be done in Triton core.

#### 1. Creating Separate Triton Models

3b. Create a copy of the `tensorrt_llm` model:

```bash
cp llama_ifb/tensorrt_llm llama_ifb/tensorrt_llm_2 -r
```

4b. Modify the `gpu_device_ids` field in the config file to specify which GPUs
should be used by each model:

```bash
sed -i 's/\${gpu_device_ids}/0,1/g' llama_ifb/tensorrt_llm/config.pbtxt
sed -i 's/\${gpu_device_ids}/2,3/g' llama_ifb/tensorrt_llm_2/config.pbtxt
sed -i 's/name: "tensorrt_llm"/name: "tensorrt_llm_2"/g' llama_ifb/tensorrt_llm_2/config.pbtxt
```

> **Note**
>
> If you want to use the ensemble or BLS models, you have to create a
> copy of the ensemble and BLS models as well and modify the "tensorrt_llm"
> model name to "tensorrt_llm_2" in the config file.

5b. Launch the server:

```bash
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --multi-model --model_repo=llama_ifb/
```

Alternatively, you can start all MPI ranks at once and avoid dynamic process spawning
by using the `--disable-spawn-processes`. The config file must specify which ranks each
model should use:

```bash
sed -i 's/\${participant_ids}/1,2/g' llama_ifb/tensorrt_llm/config.pbtxt
sed -i 's/\${participant_ids}/3,4/g' llama_ifb/tensorrt_llm_2/config.pbtxt
```

Note that rank 0 is reserved for the orchestrator rank.

```bash
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --multi-model --model_repo=llama_ifb/ --disable-spawn-processes --world_size=5
```

6b. Run the test client to measure performance:

```bash
pip3 install tritonclient[all]

# We will only benchmark the core tensorrtllm models.
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py --max-input-len 500 \
     --tensorrt-llm-model-name tensorrt_llm \
     --tensorrt-llm-model-name tensorrt_llm_2 \
     dataset --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json \
     --tokenizer-dir $HF_LLAMA_MODEL
```

7b. Kill the server:

```bash
pgrep mpirun | xargs kill
```

#### 2. Using Triton Core's Load Balancing

In order to use Triton core's load balancing for multiple instances, you can
increase the number of instances in the `instance_group` field and use the
`gpu_device_ids` parameter to specify which GPUs will be used by each model
instance.

For example, if you're running a TP=2 model on a 4-GPU system and you want
to run one instance on GPUs 0 and 1 and the other instance on GPUs 2 and 3,
you can use the following model configuration:

```
instance_group [
    {kind: KIND_CPU, count: 2}
]

parameters: {
  key: "gpu_device_ids"
  value: {
    string_value: "0,1;2,3"
  }
}
```

Please note that the number of set of GPU device ids must equal the number of instances.

### Orchestrator Mode vs Leader Mode Summary

The table below summarizes the differences between the orchestrator mode and
leader mode:

|                                   | Orchestrator Mode (Separate Models)  | Orchestrator Mode (Triton Load Balancing) |Leader Mode |
| ----------------------------------| :----------------: | :----------------: |:----------:|
| Requires Reverse Proxy            |         ❌         |           ❌        |     ✅     |
| Requires Client Changes           |         ✅         |           ❌         |     ❌     |

Orchestrator mode by default uses `MPI_Comm_Spawn` to create the child
processes. If `MPI_Comm_Spawn` is used, it is not possible to distribute
the model across multiple nodes.

It is also possible to use orchestrator mode with MPI processes that have been
pre-spawned. In order to do that, you need to set `--disable-spawn-processes`
when using the [launch_triton_server.py](../scripts/launch_triton_server.py)
script or `export TRTLLM_ORCHESTRATOR_SPAWN_PROCESSES=0`. In this mode,
it is possible to run the server across different nodes in orchestrator mode.

In order to use the orchestrator mode itself, you need to set the `--multi-model`
flag when using the [launch_triton_server.py](../scripts/launch_triton_server.py)
script or `export TRTLLM_ORCHESTRATOR=1`.
