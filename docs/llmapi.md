## End to end workflow to use the pytorch LLMAPI workflow

* Start the Triton Server Docker container:

```bash
# Replace <yy.mm> with the version of Triton you want to use.
# The command below assumes the the current directory is the
# TRT-LLM backend root git repository.

docker run --rm -ti -v `pwd`:/mnt -w /mnt -v ~/.cache/huggingface:~/.cache/huggingface --gpus all nvcr.io/nvidia/tritonserver:\<yy.mm\>-trtllm-python-py3 bash
```

* Prepare config

```bash
 cp -R tensorrt_llm/triton_backend/all_models/llmapi/ llmapi_repo/
```

Edit `llmapi_repo/tensorrt_llm/1/model.yaml` to change the model. You can either use a HuggingFace path or a local path. The following is based on `meta-llama/Llama-3.1-8B`.

This configuration file also allows you to enable CUDA graphs support and set pipeline parallelism and tensor parallelism sizes.

* Launch server

```bash
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --model_repo=llmapi_repo/
```

* Send request

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate -d '{"text_input": "The future of AI is", "max_tokens":10}' | jq
```

* Optional: include performance metrics

To retrieve detailed performance metrics per request such as KV cache usage, timing breakdowns, and speculative decoding statistics - add `"sampling_param_return_perf_metrics": true` to your request payload:

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate -d '{"text_input": "Please explain to me what is machine learning?", "max_tokens":10, "sampling_param_return_perf_metrics":true}' | jq
```

Sample response with performance metrics
```json
{
  "acceptance_rate": "0.0",
  "arrival_time_ns": "76735247746000",
  "first_scheduled_time_ns": "76735248284000",
  "first_token_time_ns": "76735374300000",
  "kv_cache_alloc_new_blocks": "1",
  "kv_cache_alloc_total_blocks": "1",
  "kv_cache_hit_rate": "0.0",
  "kv_cache_missed_block": "1",
  "kv_cache_reused_block": "0",
  "last_token_time_ns": "76736545324000",
  "model_name": "tensorrt_llm",
  "model_version": "1",
  "text_output": "Please explain to me what is machine learning? \n\nMachine learning is a field of computer science that involves the development of algorithms and models that can learn from data without being explicitly programmed. It is a",
  "total_accepted_draft_tokens": "0",
  "total_draft_tokens": "0"
}
```

`inflight_batcher_llm_client.py` is not supported yet.

* Run test on dataset

```bash
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json --max-input-len 500 --test-llmapi --model-name tensorrt_llm

[INFO] Start testing on 13 prompts.
[INFO] Functionality test succeeded.
[INFO] Warm up for benchmarking.
FLAGS.model_name: tensorrt_llm
[INFO] Start benchmarking on 13 prompts.
[INFO] Total Latency: 377.254 ms
```

* Run benchmark

```bash
 python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py --max-input-len 500 \
    --tensorrt-llm-model-name tensorrt_llm \
    --test-llmapi \
    dataset --dataset ./tensorrt_llm/triton_backend/tools/dataset/mini_cnn_eval.json \
    --tokenizer-dir meta-llama/Llama-3.1-8B

dataset
Tokenizer: Tokens per word =  1.308
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 39 prompts.
[INFO] Total Latency: 1446.623 ms
```

### Start the server on a multi-node configuration

The `srun` tool can be used to start the server in a multi-node environment:

```
srun -N 2 \
    --ntasks-per-node=8 \
    --mpi=pmix \
    --container-image=<your image> \
    --container-mounts=$(pwd)/tensorrt_llm/:/code \
    trtllm-llmapi-launch /opt/tritonserver/bin/tritonserver --model-repository llmapi_repo

```

Note: inter-node tensor parallelism is not yet supported.
