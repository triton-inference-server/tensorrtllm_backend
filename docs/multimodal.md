# End to end workflow to run a Multimodal model

### Support Matrix
The following multimodal model is supported in tensorrtllm_backend:
* BLIP2-OPT
* LLAVA

For more multimodal models supported in TensorRT-LLM, please visit [TensorRT-LLM multimodal examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal).

## Run Multimodal with single-GPU Tritonserver
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

    2-1. Clone the target model repository
    ```bash
    # For BLIP-OPT2
    export MODEL_NAME="blip2-opt-2.7b"
    git clone https://huggingface.co/Salesforce/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    # For LLAVA
    export MODEL_NAME="llava-1.5-7b-hf"
    git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```
    2-2. Build TensorRT-LLM engines
    ```bash
    export HF_MODEL_PATH=tmp/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=tmp/trt_models/${MODEL_NAME}/fp16/1-gpu
    export ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
    export VISUAL_ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/vision_encoder

    # For BLIP-OPT2
    python tensorrt_llm/examples/opt/convert_checkpoint.py --model_type blip2 \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16

    trtllm-build \
        --checkpoint_dir ${UNIFIED_CKPT_PATH} \
        --output_dir ${ENGINE_PATH} \
        --gemm_plugin float16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_seq_len 1024 \
        --max_input_len 924 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features) for BLIP2

    python tensorrt_llm/examples/multimodal/build_visual_engine.py --model_type blip2 --model_path ${HF_MODEL_PATH} --max_batch_size 8

    # For LLAVA
    python tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16

    trtllm-build \
        --checkpoint_dir ${UNIFIED_CKPT_PATH} \
        --output_dir ${ENGINE_PATH} \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 4608 # 8 (max_batch_size) * 576 (num_visual_features) for LLaVA

    python tensorrt_llm/examples/multimodal/build_visual_engine.py --model_path ${HF_MODEL_PATH} --model_type llava --max_batch_size 8

    ```

    > **NOTE**:
    >
    > `max_multimodal_len = max_batch_size * num_visual_features`, so if you change `max_batch_size`, `max_multimodal_len` **MUST** be changed accordingly.
    >
    > The built visual engines are located in `tmp/trt_engines/${MODEL_NAME}/vision_encoder`.

3. Prepare Tritonserver configs

    ```bash
    cp all_models/inflight_batcher_llm/ multimodal_ifb -r
    # Override the ensemble and creates new multimodal_encoders directories for multimodal
    cp all_models/multimodal/ensemble multimodal_ifb -r
    cp all_models/multimodal/multimodal_encoders multimodal_ifb -r

    python3 tools/fill_template.py -i multimodal_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,exclude_input_in_output:True

    python3 tools/fill_template.py -i multimodal_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${ENGINE_PATH}

    python3 tools/fill_template.py -i multimodal_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1

    python3 tools/fill_template.py -i multimodal_ifb/ensemble/config.pbtxt triton_max_batch_size:8

    python3 tools/fill_template.py -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders

    # Newly added for multimodal
    python3 tools/fill_template.py -i multimodal_ifb/multimodal_encoders/config.pbtxt triton_max_batch_size:8,visual_model_path:${VISUAL_ENGINE_PATH}
    ```
    > **NOTE**:
    >
    > You can set the `decoupled_mode` option to True to use streaming mode.
    >
    > You can set the `accumulate_tokens` option to True in streaming mode to call the postprocessing model with all accumulated tokens.

4. Launch Tritonserver

    ```bash
    python3 scripts/launch_triton_server.py --world_size 1 --model_repo=multimodal_ifb/ --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    ```

    > **NOTE**:
    > When launching the server, since the prompt_embedding_table is in GPU memory, we need to set the CUDA pool memory for inter-step communication. For example, when we have a shape of (1, 576, 4096) promp_embedding table, we would need 300MB of CUDA pool memory, so we set 30MB to have some GPU buffers. (2(fp16=>2bytes) * 576 * 4096 * 8(max_batch_size) = 18,874,368)
    >
    > Also, the tensorrt_llm initialization assumes using another GPU, we need to initialize it but not use thenm.

### Send requests
1. Send request with `decoupled_mode` set to False
    ```bash
    python tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 41.942 ms
    ```
2. Send request with `decoupled_mode` set to True
    ```bash
    python tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --streaming

    [beam 0 ]:   sing
    [beam 0 ]:  apore
    [beam 0 ]:
    [INFO] Latency: 43.441 ms
    ```
3. Send request to the `tensorrt_llm_bls` model
    ```bash
    python tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --use_bls

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 44.152 ms
    ```

4. Send request to the `tensorrt_llm_bls` model with `accumulate_tokens` set to True
    ```bash
    python tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --use_bls --streaming

    [beam 0 ]:   sing
    [beam 0 ]:   singapore
    [beam 0 ]:   singapore
    [INFO] Latency: 45.48 ms
    ```

> **NOTE**:
> Please ignore any exception thrown with the output. It's a known issue to be fixed.
>
> If there is an error associated with 'MPI_Init_thread', please do `export PMIX_MCA_gds=hash`'
### Kill the server
```bash
pkill tritonserver
```
