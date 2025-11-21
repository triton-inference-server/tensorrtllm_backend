# End to end workflow to run a Multimodal model

### Support Matrix
The following multimodal model is supported in tensorrtllm_backend:
* Whisper
* Distil-Whisper

## Run Whisper with single-GPU Tritonserver
### Tritonserver setup steps
0. Make sure that you have initialized the TRT-LLM submodule:

    ```bash
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
    git lfs install
    git submodule update --init --recursive
    ```

1. Start the Triton Server Docker container:

    1-1. If you're using Tritonserver from nvcr.io
    ```bash
    # Replace <yy.mm> with the version of Triton you want to use.
    # The command below assumes the the current directory is the
    # TRT-LLM backend root git repository.

    docker run --rm -ti --net=host -v `pwd`:/mnt -w /mnt --gpus all nvcr.io/nvidia/tritonserver:\<yy.mm\>-trtllm-python-py3 bash
    ```
    1-2. If you are using `tensorrtllm_backend` container:
    ```bash
    docker run --rm -ti --net=host -v `pwd`:/mnt -w /mnt --gpus all triton_trt_llm
    ```

2. Build the engine:

    2-1. Download the whisper models
    ```bash
    wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
    wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
    wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
    # take large-v3 model as an example
    wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
    ```
    2-2. Build TensorRT-LLM engines
    ```bash
    INFERENCE_PRECISION=float16
    MAX_BEAM_WIDTH=4
    MAX_BATCH_SIZE=64
    checkpoint_dir=tllm_checkpoint
    output_dir=whisper_large_v3_max_batch_${MAX_BATCH_SIZE}

    python3 convert_checkpoint.py --model_dir ${MODEL_DIR} --output_dir ${checkpoint_dir}

    trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
                --output_dir ${output_dir}/encoder \
                --moe_plugin disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --max_input_len 3000 --max_seq_len=3000

    trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --moe_plugin disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_seq_len 114 \
                --max_input_len 14 \
                --max_encoder_input_len 3000 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION}

    ```

    > **NOTE**:
    >
    > TensorRT-LLM also supports using [distil-whisper's](https://github.com/huggingface/distil-whisper) different models by first converting their params and weights from huggingface's naming format to [openai whisper](https://github.com/openai/whisper) naming format. You can do so by running the script [distil_whisper/convert_from_distil_whisper.py](./convert_from_distil_whisper.py).

3. Prepare Tritonserver configs

    ```bash
    cp tensorrt_llm/triton_backend/all_models/whisper/ model_repo_whisper -r
    cp tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm model_repo_whisper -r
    wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
    wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

    BACKEND=tensorrtllm
    DECOUPLED_MODE=false
    DECODER_ENGINE_PATH=${output_dir}/decoder
    ENCODER_ENGINE_PATH=${output_dir}/encoder
    MAX_TOKENS_IN_KV_CACHE=24000
    BATCHING_STRATEGY=inflight_fused_batching
    KV_CACHE_FREE_GPU_MEM_FRACTION=0.5
    EXCLUDE_INPUT_IN_OUTPUT=True
    TRITON_MAX_BATCH_SIZE=8
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MAX_BEAM_WIDTH=1
    MAX_QUEUE_SIZE="0"
    ENABLE_KV_CACHE_REUSE=false
    ENABLE_CHUNKED_CONTEXT=false
    CROSS_KV_CACHE_FRACTION="0.5"
    n_mels=128
    zero_pad=false

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i model_repo_whisper/tensorrt_llm/config.pbtxt triton_backend:${BACKEND},engine_dir:${DECODER_ENGINE_PATH},encoder_engine_dir:${ENCODER_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE},enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACC},cross_kv_cache_fraction:${CROSS_KV_CACHE_FRACTION},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i model_repo_whisper/whisper_bls/config.pbtxt engine_dir:${ENCODER_ENGINE_PATH},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE}
    ```
    > **NOTE**:
    >
    > TODO: You can set the `decoupled_mode` option to True to use streaming mode.

4. Launch Tritonserver

    ```bash
    python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 1 --model_repo=model_repo_whisper/ --tensorrt_llm_model_name tensorrt_llm,whisper_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    ```

### Send requests
1. Send request with a single audio file
    ```bash
    wget -nc https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
    # Test non-streaming
    python3 tensorrt_llm/triton_backend/whisper/client.py --audio-path 1221-135766-0002.wav
    ```
2. Send requests with a whole audio dataset
   ```bash
    git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
    cd Triton-ASR-Client
    num_task=16
    python3 tensorrt_llm/triton_backend/whisper/client.py \
        --server-addr localhost \
        --model-name whisper_bls \
        --num-tasks $num_task \
        --text-prompt "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
        --manifest-dir ./datasets/aishell1_test \
        --compute-cer
    ```
### Kill the server
```bash
pkill tritonserver
```
