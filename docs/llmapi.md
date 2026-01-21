# End-to-End Workflow for the LLMAPI Backend

The LLMAPI backend provides a simplified way to deploy TensorRT-LLM models with Triton Inference Server. It uses TensorRT-LLM's high-level `LLM` class which automatically handles model loading from HuggingFace, supports the PyTorch backend, and enables features like tensor parallelism, streaming, and speculative decoding.

## Quick Start

### Start the Triton Server Docker Container

```bash
# Replace <yy.mm> with the version of Triton you want to use.
# The command below assumes the current directory is the TRT-LLM backend root git repository.

docker run --rm -ti \
    -v $(pwd):/mnt -w /mnt \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --gpus all \
    nvcr.io/nvidia/tritonserver:<yy.mm>-trtllm-python-py3 bash
```

### Prepare the Model Repository

```bash
cp -R tensorrt_llm/triton_backend/all_models/llmapi/ llmapi_repo/
```

### Configure the Model

Edit `llmapi_repo/tensorrt_llm/1/model.yaml` to specify your model and settings. You can use either a HuggingFace model ID or a local path.

**Basic configuration example:**

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

triton_config:
  max_batch_size: 0
  decoupled: False
```

**Multi-GPU configuration:**

```yaml
model: meta-llama/Llama-3.1-70B-Instruct
backend: pytorch

tensor_parallel_size: 8
pipeline_parallel_size: 1

triton_config:
  max_batch_size: 0
  decoupled: True
```

The `triton_config` section controls Triton-specific settings:
- `max_batch_size: 0` - Batching is handled by the LLMAPI, not Triton
- `decoupled: True` - Required for streaming mode

### Launch the Server

```bash
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --model_repo=llmapi_repo/
```

### Send Requests

**Basic request:**

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "The future of AI is", "sampling_param_max_tokens": 100}' | jq
```

**With sampling parameters:**

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "Hello, my name is", "sampling_param_max_tokens": 100, "sampling_param_temperature": 0.8, "sampling_param_top_p": 0.95}' | jq
```

**Streaming request (requires `decoupled: True` in model.yaml):**

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate_stream \
    -d '{"text_input": "Write a poem about", "sampling_param_max_tokens": 100}' | sed 's/^data: //' | jq
```

## Configuration Options

The `model.yaml` file accepts parameters from TensorRT-LLM's `LLM` class. Common options include:

| Parameter | Description |
|-----------|-------------|
| `model` | HuggingFace model ID or local path (required) |
| `backend` | Backend to use (`pytorch` recommended) |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `pipeline_parallel_size` | Number of stages for pipeline parallelism |
| `max_batch_size` | Maximum batch size for inference |
| `max_num_tokens` | Maximum tokens per iteration (default: 8192) |
| `trust_remote_code` | Set to `true` for models requiring custom code |
| `enable_chunked_prefill` | Enable chunked prefill for better throughput |

**KV cache configuration:**

```yaml
kv_cache_config:
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.85
```

**CUDA graphs for faster decode (PyTorch backend):**

```yaml
cuda_graph_config:
  max_batch_size: 16
```

**Speculative decoding:**

```yaml
speculative_config:
  decoding_type: Lookahead
  max_window_size: 4
  max_ngram_size: 3
```

## Performance Metrics

To retrieve detailed performance metrics per request (KV cache usage, timing breakdowns, speculative decoding statistics), add `"sampling_param_return_perf_metrics": true` to your request:

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "What is machine learning?", "sampling_param_max_tokens": 50, "sampling_param_return_perf_metrics": true}' | jq
```

Sample response with metrics:

```json
{
  "text_output": "Machine learning is a field of...",
  "kv_cache_hit_rate": "0.625",
  "kv_cache_reused_block": "5",
  "arrival_time_ns": "76735247746000",
  "first_token_time_ns": "76735374300000",
  "last_token_time_ns": "76736545324000",
  "acceptance_rate": "0.0"
}
```

## Multi-Node Configuration

For multi-node deployments, use `srun` with `trtllm-llmapi-launch`:

```bash
srun -N 2 \
    --ntasks-per-node=8 \
    --mpi=pmix \
    --container-image=<your-image> \
    --container-mounts=$(pwd)/tensorrt_llm/:/code \
    trtllm-llmapi-launch /opt/tritonserver/bin/tritonserver --model-repository llmapi_repo
```

Configure pipeline parallelism across nodes in `model.yaml`:

```yaml
model: meta-llama/Llama-3.1-405B-Instruct
tensor_parallel_size: 8
pipeline_parallel_size: 2
gpus_per_node: 8
```

**Note:** Inter-node tensor parallelism is not supported. Use pipeline parallelism for multi-node deployments.

## Testing

Run the end-to-end test:

```bash
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py \
    --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json \
    --max-input-len 500 \
    --test-llmapi \
    --model-name tensorrt_llm
```

Run a benchmark:

```bash
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py \
    --max-input-len 500 \
    --tensorrt-llm-model-name tensorrt_llm \
    --test-llmapi \
    dataset --dataset ./tensorrt_llm/triton_backend/tools/dataset/mini_cnn_eval.json \
    --tokenizer-dir meta-llama/Llama-3.1-8B
```

## Notes

- Set `triton_config.decoupled: True` in model.yaml to enable streaming mode.
- The LLMAPI automatically handles model loading and optimization - no pre-built engines required.
- Available request parameters are listed in the `config.pbtxt` file (sampling_param_max_tokens, sampling_param_temperature, sampling_param_top_p, etc.).
