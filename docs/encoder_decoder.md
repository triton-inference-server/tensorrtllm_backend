# End to end workflow to run an Encoder-Decoder model

### Support Matrix
For the specific models supported by encoder-decoder family, please visit [TensorRT-LLM encoder-decoder examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec#encoder-decoder-model-support). The following two model types are supported:
* T5
* BART

## Run Encoder-Decoder with single-GPU Tritonserver
### Tritonserver setup steps
0. Make sure that you have initialized the TRT-LLM submodule:

    ```bash
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
    git lfs install
    git submodule update --init --recursive
    ```

1. (Optional) Download the target model from HuggingFace:

    ```bash
    huggingface-cli login

    huggingface-cli download google-t5/t5-small
    ```

    > **NOTE**
    >
    > Make sure that you have access to https://huggingface.co/google-t5/t5-small.

2. Start the Triton Server Docker container:
    > **NOTE**
    >
    > The current Tritonserver version (24.05) is not yet updated to the latest version of TensorRT-LLM that has enc-dec support. Please follow the [build the Docker container guide](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#build-the-docker-container) to build the up-to-date `tensorrtllm_backend` container.

    2-1. If you're using Tritonserver from nvcr.io
    ```bash
    # Replace <yy.mm> with the version of Triton you want to use.
    # The command below assumes the the current directory is the
    # TRT-LLM backend root git repository.

    docker run --rm -ti -v `pwd`:/mnt -w /mnt -v ~/.cache/huggingface:~/.cache/huggingface --gpus all nvcr.io/nvidia/tritonserver:\<yy.mm\>-trtllm-python-py3 bash
    ```
    2-2. If you are using `tensorrtllm_backend` container:
    ```bash
    docker run --rm -ti -v `pwd`:/mnt -w /mnt -v ~/.cache/huggingface:~/.cache/huggingface --gpus all triton_trt_llm
    ```

3. Build the engine:

    3-1. Clone the target model repository if you didn't do step 1
    ```bash
    git clone https://huggingface.co/google-t5/t5-small /tmp/hf_models/t5-small
    ```
    3-2. Build TensorRT-LLM engines
    ```bash
    # Replace 'HF_MODEL_PATH' with another path if you didn't download the model from step 1
    # or you're not using HuggingFace.
    export MODEL_NAME=t5-small
    export MODEL_TYPE=t5 # or bart
    export HF_MODEL_PATH=/tmp/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=/tmp/ckpt/${MODEL_NAME}
    export ENGINE_PATH=/tmp/engines/${MODEL_NAME}
    python tensorrt_llm/examples/enc_dec/convert_checkpoint.py \
        --model_type ${MODEL_TYPE} \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/encoder \
        --output_dir ${ENGINE_PATH}/encoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_batch_size 64 \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
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
    <!-- > **NOTE**
    >
    > For bart family, remove `--context_fmha disable`. Please refer to [TensorRT-LLM enc-dec](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec#build-tensorrt-engines) for details. -->

4. Prepare Tritonserver configs <a id="prepare-tritonserver-configs"></a>

    ```bash
    cp all_models/inflight_batcher_llm/ enc_dec_ifb -r

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH}/decoder,encoder_engine_dir:${ENGINE_PATH}/encoder,max_tokens_in_paged_kv_cache:4096,max_attention_window_size:4096,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,max_queue_size:0

    python3 tools/fill_template.py -i enc_dec_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,preprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,postprocessing_instance_count:1

    python3 tools/fill_template.py -i enc_dec_ifb/ensemble/config.pbtxt triton_max_batch_size:64

    python3 tools/fill_template.py -i enc_dec_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
    ```

5. Launch Tritonserver

    ```bash
    pip install SentencePiece
    python3 scripts/launch_triton_server.py --world_size 1 --model_repo=enc_dec_ifb/
    ```

### Send requests
1. Send request with CURL
    ```bash
    curl -X POST localhost:8000/v2/models/ensemble/generate -d "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": \"\", \"stop_words\": \"\"}"

    {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":":::: (CNN): (CNN): (CNN) the Hit & Run Tour to the U.S. for the first time. the Hit & Run Tour will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Tickets will go on sale Monday, March 9 at 10 a.m. local time."}
    ```

2. Send request with bad_words and stop_words

    After applying the `stop_words` and `bad_words`, the output avoids the bad words and stops at the first generated stop word.
    ```bash
    curl -X POST localhost:8000/v2/models/ensemble/generate -d "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": [\"drummer\", \"hometown\"], \"stop_words\": [\"Tickets\", \"sale\"]}"

    {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":":::: (CNN): (CNN): (CNN) the Hit & Run Tour to the U.S. for the first time. the Hit & Run Tour will take place in Louisville, Kentucky, the home of 3rdEyeGirl's Hannah Welton."}
    ```

3. Send request by `inflight_batcher_llm_client.py`
    ```bash
    python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 1024 --tokenizer-dir ${HF_MODEL_PATH}

    ========
    Using pad_id:  0
    Using end_id:  1
    Input sequence:  [12896, 16, 3457, 18, 11535, 1410, 6, 264, 7975, 4252, 38, 3, 9, 1]
    [TensorRT-LLM][WARNING] decoder_input_ids is not present in the request for encoder-decoder model. The decoder input tokens will be set to [padId]
    Got completed request
    Input: Born in north-east France, Soyer trained as a
    Output beam 0: Universität in Paris.
    Output sequence:  [32099, 16, 3457, 18, 11535, 1410, 6, 264, 7975, 3, 8637, 3930, 46, 74, 16086, 16, 1919, 5]
    ```
    > **NOTE**
    >
    > Please ignore any exception thrown with the output. It's a known issue to be fixed.

4. Run test on dataset

    ```
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500

    [INFO] Start testing on 13 prompts.
    [INFO] Functionality test succeed.
    [INFO] Warm up for benchmarking.
    [INFO] Start benchmarking on 13 prompts.
    [INFO] Total Latency: 155.756 ms
    ```

5. Run several requests at the same time

    ```bash
    echo "{\"text_input\": \"Summarize the following news article: (CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.\", \"max_tokens\": 1024, \"bad_words\": [\"drummer\", \"hometown\"], \"stop_words\": [\"Tickets\", \"sale\"]}" > tmp.txt
    printf '%s\n' {1..20} | xargs -I % -P 20 curl -X POST localhost:8000/v2/models/ensemble/generate -d @tmp.txt
    ```

## Running multiple instances of encoder-decoder model on multiple GPU

In this section, we demonstrate how you can use
[Leader Mode](../README.md#leader-mode) for running multiple instances of a encoder-decoder model on different GPUs.

For this section, let's assume that we have four GPUs and the CUDA device ids
are 0, 1, 2, and 3.  We will be launching two instances of the T5-small model
with tensor parallelism 2 (TP=2). The first instance will run on GPUs 0 and 1
and the second instance will run on GPUs 2 and 3. We will launch two separate `mpirun` commands to launch two separate Triton servers, one for each GPU (4 Triton Server instances in total). We also need to use a reverse proxy in front of them to load balance the requests between the servers.

[Orchestrator Mode](../README.md#orchestrator-mode) currently not supported.


### Tritonserver setup steps
1. Build the model, but add `--tp_size 2` when converting checkpoints. The rest are the same as [Tritonserver Single-GPU Setup
](#tritonserver-single-gpu-setup).
    ```
    export MODEL_NAME=t5-small
    export MODEL_TYPE=t5 # or bart
    export HF_MODEL_PATH=/tmp/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=/tmp/ckpt/${MODEL_NAME}-2tp-2gpu
    export ENGINE_PATH=/tmp/engines/${MODEL_NAME}-2tp-2gpu

    python tensorrt_llm/examples/enc_dec/convert_checkpoint.py \
        --model_type ${MODEL_TYPE} \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16 \
        --tp_size 2

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/encoder \
        --output_dir ${ENGINE_PATH}/encoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_batch_size 64 \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
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
2. Setup Tritonserver config with the same commands in [step 4](#prepare-tritonserver-configs) of [Tritonserver Single-GPU Setup
](#tritonserver-single-gpu-setup).


3. Launch the servers:

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/launch_triton_server.py --world_size 2 --model_repo=enc_dec_ifb/ --http_port 8000 --grpc_port 8001 --metrics_port 8004
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/launch_triton_server.py --world_size 2 --model_repo=enc_dec_ifb/ --http_port 8002 --grpc_port 8003 --metrics_port 8005
    ```

4. Install NGINX:

    ```bash
    apt update
    apt install nginx -y
    ```

5. Setup the NGINX configuration and store it in `/etc/nginx/sites-available/tritonserver`:

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

6. Create a symlink and restart NGINX to enable the configuration:

    ```bash
    ln -s /etc/nginx/sites-available/tritonserver /etc/nginx/sites-enabled/tritonserver
    service nginx restart
    ```

### Send the request

1. Run test on dataset

    ```bash
    # Test the load on all the servers
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8080

    # Test the load on one of the servers
    python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500 -u localhost:8000
    ```

### Kill the server
```bash
pgrep mpirun | xargs kill
```
