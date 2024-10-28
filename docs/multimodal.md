# End to end workflow to run a Multimodal model

### Support Matrix
The following multimodal model is supported in tensorrtllm_backend:
* BLIP2-OPT
* LLAVA
* VILA

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

    # For VILA
    pip install -r all_models/multimodal/requirements-vila.txt

    export MODEL_NAME="vila1.5-3b"
    git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    export VILA_PATH="tmp/hf_models/VILA"
    git clone https://github.com/Efficient-Large-Model/VILA.git ${VILA_PATH}
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

    # For VILA
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
        --max_multimodal_len 6272 # 8 (max_batch_size) * 196 (num_visual_features) * 4 (max_num_images_per_request)

    python tensorrt_llm/examples/multimodal/build_visual_engine.py --model_path ${HF_MODEL_PATH} --model_type vila --vila_path ${VILA_PATH} --max_batch_size 32 #max_batch_size * max_num_images_per_request since vila support multiple images inference

    ```

    > **NOTE**:
    >
    > `max_multimodal_len = max_batch_size * num_visual_features`, so if you change `max_batch_size`, `max_multimodal_len` **MUST** be changed accordingly.
    > For multi-image inference, where a single request could contain multiple images, `max_multimodal_len = max_batch_size * num_visual_features * max_num_images_per_request`
    >
    > The built visual engines are located in `tmp/trt_engines/${MODEL_NAME}/vision_encoder`.

3. Prepare Tritonserver configs

    ```bash
    cp all_models/inflight_batcher_llm/ multimodal_ifb -r
    # Override the ensemble and creates new multimodal_encoders directories for multimodal
    cp all_models/multimodal/ensemble multimodal_ifb -r
    cp all_models/multimodal/multimodal_encoders multimodal_ifb -r

    python3 tools/fill_template.py -i multimodal_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False

    python3 tools/fill_template.py -i multimodal_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1

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
    >
    > You can set the `enable_kv_cache_reuse` option to True to enable kv cache reuse. Requests with the same image/prompt table/input tokens will reuse the KV cache, which will help reduce latency. The specific performance improvement depends on the length of reuse.
    >
    > You can set the `max_num_images` to the max number of images per request. The value should be the same as the `max_num_images_per_request` value used at build the engine step above.

4. Launch Tritonserver

    ```bash
    python3 scripts/launch_triton_server.py --world_size 1 --model_repo=multimodal_ifb/ --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    ```

    > **NOTE**:
    > When launching the server, since the prompt_embedding_table is in GPU memory, we need to set the CUDA pool memory for inter-step communication. For example, when we have a shape of (1, 576, 4096) promp_embedding table, we would need 300MB of CUDA pool memory, so we set 30MB to have some GPU buffers. (2(fp16=>2bytes) * 576 * 4096 * 8(max_batch_size) = 18,874,368)
    >
    > Also, the tensorrt_llm initialization assumes using another GPU, we need to initialize it but not use them.

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

5. Send request with `enable_kv_cache_reuse` set to True
    ```bash
    python tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --prompt_table_extra_id ${id}

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 42.514 ms
    ```
6. Send request with multiple images per request
    ```bash
    wget -O av.png https://raw.githubusercontent.com/Efficient-Large-Model/VILA/main/demo_images/av.png

    python tools/multimodal/client.py --text '<image>\n<image>\n Please elaborate what you see in the images?' --image av.png,'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 68 --model_type vila --hf_model_dir ${HF_MODEL_PATH}

    [beam 0 ]:
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:  \n \n Please elaborate what you see in the images? ASSISTANT: The first image shows a busy street scene with a car driving through a crosswalk, surrounded by pedestrians and traffic lights. The second image captures a beautiful sunset with the iconic Merlion statue spouting water into the bay, with the Singapore Flyer and the city skyline in the background.

    [INFO] Latency: 403.879 ms
    ```

> **NOTE**:
> Please ignore any exception thrown with the output. It's a known issue to be fixed.
>
> If there is an error associated with 'MPI_Init_thread', please do `export PMIX_MCA_gds=hash`'
>
> When `enable_kv_cache_reuse` is set to true, the `prompt_table_extra_id` must be specified in the requests. The `prompt_table_extra_id` is a unique identifier representing the image (or prompt table), the same image uses the same id. The data type is `uint64`, and the minimum value is 1.

### Kill the server
```bash
pkill tritonserver
```
