# End to end workflow to run an Encoder-Decoder model

### Support Matrix
For the specific models supported by encoder-decoder family, please visit [TensorRT-LLM encoder-decoder examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec#encoder-decoder-model-support). The following two model types are supported:
* T5
* BART

## Run Encoder-Decoder with Tritonserver
### Tritonserver setup steps

#### 1. Make sure that you have initialized the TRT-LLM submodule:

```
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
    git lfs install
    git submodule update --init --recursive
```

#### 2. Start the Triton Server Docker container within `tensorrtllm_backend` repo:

If you're using [Triton TRT-LLM NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)

```
    # Replace <yy.mm> with the version of Triton you want to use. Here using 24.08.
    # The commands below assumes the the current directory is the
    # TRT-LLM backend root git repository.

    docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g `pwd`:/workspace -w /workspace nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 bash
```

If [building your own TensorRT-LLM Backend container](https://github.com/triton-inference-server/tensorrtllm_backend#option-2-build-via-docker) then you can run the `tensorrtllm_backend` container:

```
    docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g `pwd`:/workspace -w /workspace triton_trt_llm bash
```

#### 3. Build the engines:

Clone the target model repository from HuggingFace. Here we use [T5-small model](https://huggingface.co/google-t5/t5-small) as example but you can also follow the same steps for BART model.


    git lfs install
    git clone https://huggingface.co/google-t5/t5-small /workspace/hf_models/t5-small


Build TensorRT-LLM engines.

```
    export MODEL_NAME=t5-small # or bart-base
    export MODEL_TYPE=t5 # or bart
    export HF_MODEL_PATH=/workspace/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=/workspace/ckpt/${MODEL_NAME}
    export ENGINE_PATH=/workspace/engines/${MODEL_NAME}
    export INFERENCE_PRECISION=float16
    export TP_SIZE=1
    export MAX_BEAM_WIDTH=1
    export MAX_BATCH_SIZE=8
    export INPUT_LEN=1024
    export OUTPUT_LEN=201

    python3 tensorrt_llm/examples/enc_dec/convert_checkpoint.py \
    --model_type ${MODEL_TYPE} \
    --model_dir ${HF_MODEL_PATH} \
    --output_dir ${UNIFIED_CKPT_PATH} \
    --dtype ${INFERENCE_PRECISION} \
    --tp_size ${TP_SIZE}

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/encoder \
    --output_dir ${ENGINE_PATH}/encoder \
    --kv_cache_type disabled \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_beam_width ${MAX_BEAM_WIDTH} \
    --max_input_len ${INPUT_LEN} \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --gemm_plugin ${INFERENCE_PRECISION} \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --gpt_attention_plugin ${INFERENCE_PRECISION} \
    --context_fmha disable # remove for BART

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/decoder \
    --output_dir ${ENGINE_PATH}/decoder \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_beam_width ${MAX_BEAM_WIDTH} \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --gemm_plugin ${INFERENCE_PRECISION} \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --gpt_attention_plugin ${INFERENCE_PRECISION} \
    --max_input_len 1 \
    --max_encoder_input_len ${INPUT_LEN} \
    --max_seq_len ${OUTPUT_LEN} \
    --context_fmha disable # remove for BART
```

> **NOTE**
>
> If you want to build multi-GPU engine using Tensor Parallelism then you can set `--tp_size` in convert_checkpoint.py. For example, for TP=2 on 2-GPU you can set `--tp_size=2`. If you want to use beam search then set `--max_beam_width` to higher value than 1. The `--max_input_len` in encoder trtllm-build controls the model input length and should be same as `--max_encoder_input_len` in decoder trtllm-build. Additionally, to control the model output len you should set `--max_seq_len` in decoder trtllm-build to `desired output length + 1`. It is also advisable to tune [`--max_num_tokens`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md#max_num_tokens) as the default value of 8192 might be too large or too small depending on your input, output len and use-cases. For BART family models, make sure to remove `--context_fmha disable` from both encoder and decoder trtllm-build commands. Please refer to [TensorRT-LLM enc-dec example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec#build-tensorrt-engines) for more details.

#### 4. Prepare Tritonserver configs <a id="prepare-tritonserver-configs"></a>

```
    cp all_models/inflight_batcher_llm/ enc_dec_ifb -r

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:${MAX_BEAM_WIDTH},engine_dir:${ENGINE_PATH}/decoder,encoder_engine_dir:${ENGINE_PATH}/encoder,kv_cache_free_gpu_mem_fraction:0.45,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,max_queue_size:0

    python3 tools/fill_template.py -i enc_dec_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False

```

> **NOTE**
>
> Currently, encoder-decoder models don't support running with chunked context.

#### 5. Launch Tritonserver

```
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=enc_dec_ifb/
```

### Send requests
#### 1. Send request with CURL

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": \"\", \"stop_words\": \"\"}"

    {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":":::: (CNN): (CNN): (CNN) the Hit & Run Tour to the U.S. for the first time. the Hit & Run Tour will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Tickets will go on sale Monday, March 9 at 10 a.m. local time."}
```

#### 2. Send request with `bad_words` and `stop_words`

After applying the `stop_words` and `bad_words`, the output avoids the bad words and stops at the first generated stop word.

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": [\"drummer\", \"hometown\"], \"stop_words\": [\"Tickets\", \"sale\"]}"

    {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":":::: (CNN): (CNN): (CNN) the Hit & Run Tour to the U.S. for the first time. the Hit & Run Tour will take place in Louisville, Kentucky, the home of 3rdEyeGirl's Hannah Welton."}
```

#### 3. Send request by `inflight_batcher_llm_client.py`
If not already installed, install `tritonclient`

```
    pip install tritonclient[all]
    python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --text "translate English to German: This is good" --request-output-len 200 --exclude-input-in-output --tokenizer-dir ${HF_MODEL_PATH} --beam-width ${MAX_BEAM_WIDTH}

    ========
    Using pad_id:  0
    Using end_id:  1
    Input sequence:  [13959, 1566, 12, 2968, 10, 100, 19, 207, 1]
    [TensorRT-LLM][WARNING] decoder_input_ids is not present in the request for encoder-decoder model. The decoder input tokens will be set to [padId]
    Got completed request
    Input: translate English to German: This is good
    Output beam 0: Das is gut.
    Output sequence:  [644, 229, 1806, 5]
```

> **NOTE**
>
> Please ignore any exception thrown with the output. It's a known issue to be fixed.

#### 4. Run test on dataset

```
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500

    [INFO] Start testing on 13 prompts.
    [INFO] Functionality test succeed.
    [INFO] Warm up for benchmarking.
    [INFO] Start benchmarking on 13 prompts.
    [INFO] Total Latency: 155.756 ms
```

#### 5. Run several requests at the same time

```
echo "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": [\"drummer\", \"hometown\"], \"stop_words\": [\"Tickets\", \"sale\"]}" > tmp.txt

printf '%s\n' {1..20} | xargs -I % -P 20 curl -X POST localhost:8000/v2/models/ensemble/generate -d @tmp.txt
```
#### 6. Evaluating performance with Gen-AI Perf

Gen-AI Perf is a command line tool for measuring the throughput and latency of generative AI models as served through an inference server. You can read more about installing Gen-AI Perf [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html#installation).

To use Gen-AI Perf, run the following command:

```
genai-perf profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --tokenizer ${HF_MODEL_PATH} \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

You should expect an output that looks like this (the output below was obtained on A100-80GB with TRT-LLM v0.12):

```                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃              Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│   Request latency (ms) │  80.92 │  78.84 │ 323.55 │  85.14 │  79.90 │  79.64 │
│ Output sequence length │  95.83 │  65.00 │ 100.00 │ 100.00 │  99.00 │  98.00 │
│  Input sequence length │ 200.01 │ 200.00 │ 201.00 │ 200.00 │ 200.00 │ 200.00 │
└────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1182.70
Request throughput (per sec): 12.34
```

#### 7. Run with decoupled mode (streaming)

To enable streaming, we set `decoupled_mode:True` in config.pbtxt of `tensorrt_llm` and `tensorrt_llm_bls` model (if you are using BLS instead of ensemble).

```
    cp all_models/inflight_batcher_llm/ enc_dec_ifb -r

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,max_beam_width:${MAX_BEAM_WIDTH},engine_dir:${ENGINE_PATH}/decoder,encoder_engine_dir:${ENGINE_PATH}/encoder,kv_cache_free_gpu_mem_fraction:0.45,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,max_queue_size:0

    python3 tools/fill_template.py -i enc_dec_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False

```

We launch Tritonserver

```
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=enc_dec_ifb/
```

Then send request by `inflight_batcher_llm_client.py`

```
pip install tritonclient[all]
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --text "translate English to German: This is good" --request-output-len 200 --exclude-input-in-output --tokenizer-dir ${HF_MODEL_PATH} --beam-width ${MAX_BEAM_WIDTH} --streaming
```

To use Gen-AI Perf to benchmark streaming/decoupled mode, run the following command:

```
genai-perf profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --tokenizer ${HF_MODEL_PATH} \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

You should see output like this (the output below was obtained on A100-80GB with TRT-LLM v0.12)

```
                                   LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   4.69 │   3.99 │  14.05 │   5.70 │   5.04 │   4.76 │
│ Inter token latency (ms) │   0.63 │   0.38 │   1.04 │   0.98 │   0.70 │   0.66 │
│     Request latency (ms) │  75.32 │  46.34 │ 114.27 │  90.35 │  79.27 │  79.11 │
│   Output sequence length │ 116.50 │  58.00 │ 197.00 │ 197.00 │ 132.00 │ 128.00 │
│    Input sequence length │ 200.01 │ 200.00 │ 201.00 │ 200.10 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1542.81
Request throughput (per sec): 13.24
```

## Running multiple instances of encoder-decoder model on multiple GPUs

In this section, we demonstrate how you can use
[Leader Mode](../README.md#leader-mode) for running multiple instances of a encoder-decoder model on different GPUs.

For this section, let's assume that we have four GPUs and the CUDA device ids
are 0, 1, 2, and 3.  We will be launching two instances of the T5-small model
with tensor parallelism 2 (TP=2). The first instance will run on GPUs 0 and 1
and the second instance will run on GPUs 2 and 3. We will launch two separate `mpirun` commands to launch two separate Triton servers, one for each GPU (4 Triton Server instances in total). We also need to use a reverse proxy in front of them to load balance the requests between the servers.

[Orchestrator Mode](../README.md#orchestrator-mode) currently not supported.


### Triton setup steps
1. Build the model, but add `--tp_size 2` when converting checkpoints. The rest of the steps are the same as [Tritonserver Setup
](#Tritonserver-setup-steps).

```
    export MODEL_NAME=t5-small
    export MODEL_TYPE=t5 # or bart
    export HF_MODEL_PATH=/workspace/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=/workspace/ckpt/${MODEL_NAME}-2tp-2gpu
    export ENGINE_PATH=/workspace/engines/${MODEL_NAME}-2tp-2gpu

    python tensorrt_llm/examples/enc_dec/convert_checkpoint.py \
        --model_type ${MODEL_TYPE} \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16 \
        --tp_size 2

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/encoder \
        --output_dir ${ENGINE_PATH}/encoder \
        --kv_cache_type disabled \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_batch_size 64 \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
        --max_input_len 2048 \
        --context_fmha disable

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/decoder \
        --output_dir ${ENGINE_PATH}/decoder \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_batch_size 64 \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
        --context_fmha disable \
        --max_input_len 1 \
        --max_encoder_input_len 2048
```

3. Setup Tritonserver config with the same commands in [step 4](#prepare-tritonserver-configs) above.

4. Launch the servers:

```
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/launch_triton_server.py --world_size 2 --model_repo=enc_dec_ifb/ --http_port 8000 --grpc_port 8001 --metrics_port 8004
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/launch_triton_server.py --world_size 2 --model_repo=enc_dec_ifb/ --http_port 8002 --grpc_port 8003 --metrics_port 8005
```

4. Install NGINX:

```
    apt update
    apt install nginx -y
```

5. Setup the NGINX configuration and store it in `/etc/nginx/sites-available/tritonserver`:

```
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

6. Create a symlink and restart NGINX to enable the configuration:

```
    ln -s /etc/nginx/sites-available/tritonserver /etc/nginx/sites-enabled/tritonserver
    service nginx restart
```

### Send the request

1. Run test on dataset

```
    # Test the load on all the servers
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8080

    # Test the load on one of the servers
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8000
```

### Kill the server
```
pgrep mpirun | xargs kill
```
