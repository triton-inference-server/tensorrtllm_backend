# End to end workflow to run a Multimodal model

### Support Matrix
The following multimodal model is supported in tensorrtllm_backend:
* BLIP2-OPT
* LLAVA
* VILA
* LLaVA OneVision
* MLLAMA
* Qwen2-VL

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

    # For LLaVA OneVision
    pip install -r all_models/multimodal/requirements-llava-onevision.txt

    export MODEL_NAME="llava-onevision-qwen2-7b-ov-hf"
    git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    # For MLLAMA
    pip install -r all_models/multimodal/requirements-mllama.txt

    export MODEL_NAME="Llama-3.2-11B-Vision"
    git clone https://huggingface.co/meta-llama/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    # For Qwen2-VL
    pip install -r all_models/multimodal/requirements-qwen2vl.txt

    export MODEL_NAME="Qwen2-VL-7B-Instruct"
    git clone https://huggingface.co/Qwen/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    export
    ```
    2-2. Build TensorRT-LLM engines
    ```bash
    export HF_MODEL_PATH=tmp/hf_models/${MODEL_NAME}
    export UNIFIED_CKPT_PATH=tmp/trt_models/${MODEL_NAME}/fp16/1-gpu
    export ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
    export MULTIMODAL_ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/multimodal_encoder

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
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_multimodal_features) for BLIP2

    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_type blip2 --model_path ${HF_MODEL_PATH} --max_batch_size 8

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
        --max_multimodal_len 4608 # 8 (max_batch_size) * 576 (num_multimodal_features) for LLaVA

    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_path ${HF_MODEL_PATH} --model_type llava --max_batch_size 8

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
        --max_multimodal_len 6272 # 8 (max_batch_size) * 196 (num_multimodal_features) * 4 (max_num_images_per_request)

    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_path ${HF_MODEL_PATH} --model_type vila --vila_path ${VILA_PATH} --max_batch_size 32 #max_batch_size * max_num_images_per_request since vila support multiple images inference

    # For LLaVA OneVision
    python tensorrt_llm/examples/qwen/convert_checkpoint.py \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16

    trtllm-build \
        --checkpoint_dir ${UNIFIED_CKPT_PATH} \
        --output_dir ${ENGINE_PATH} \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len  7500 \
        --max_seq_len  7600 \
        --max_multimodal_len 7300 # max_batch_size * num_multimodal_features(depends on the image size or the specified video num frame)

    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_path ${HF_MODEL_PATH} --model_type llava_onevision --max_batch_size 16 # max_batch_size * patch for image or frame for video

    # For MLLAMA
    python tensorrt_llm/examples/mllama/convert_checkpoint.py \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype bfloat16

    trtllm-build \
    --checkpoint_dir ${UNIFIED_CKPT_PATH} \
    --output_dir ${ENGINE_PATH} \
    --gemm_plugin auto \
    --max_batch_size 8 \
    --max_seq_len 2048 \
    --max_num_tokens 4096 \
    --max_encoder_input_len 6404

    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_path ${HF_MODEL_PATH} --model_type mllama --output_dir ${MULTIMODAL_ENGINE_PATH} --max_batch_size 8 #max_batch_size * max_num_images_per_request

    # For Qwen2-VL
    python3 ../qwen/convert_checkpoint.py \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16

    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
        --output_dir ${ENGINE_PATH} \
        --gemm_plugin=float16 \
        --gpt_attention_plugin=float16 \
        --max_batch_size 4 \
        --max_input_len 2048 \
        --max_seq_len 3072 \
        --max_multimodal_len 1296 #(max_batch_size) * 324 (num_multimodal_features), this's for image_shape=[504,504]

    python build_multimodal_engine.py --model_type qwen2_vl --model_path tmp/hf_models/${MODEL_NAME} --output_dir ${MULTIMODAL_ENGINE_PATH}
    ```

    > **NOTE**:
    >
    > `max_multimodal_len = max_batch_size * num_multimodal_features`, so if you change `max_batch_size`, `max_multimodal_len` **MUST** be changed accordingly.
    > For multi-image inference, where a single request could contain multiple images, `max_multimodal_len = max_batch_size * num_multimodal_features * max_num_images_per_request`
    >
    > The built visual engines are located in `tmp/trt_engines/${MODEL_NAME}/multimodal_encoder`.

3. Prepare Tritonserver configs

    ```bash
    cp tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/ multimodal_ifb -r
    # Override the ensemble and creates new multimodal_encoders directories for multimodal
    cp tensorrt_llm/triton_backend/all_models/multimodal/ensemble multimodal_ifb -r
    cp tensorrt_llm/triton_backend/all_models/multimodal/multimodal_encoders multimodal_ifb -r

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.5

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1,max_queue_delay_microseconds:20000

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/ensemble/config.pbtxt triton_max_batch_size:8,logits_datatype:TYPE_FP32

    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

    # Newly added for multimodal
    python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/multimodal_encoders/config.pbtxt triton_max_batch_size:8,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH},max_queue_delay_microseconds:20000
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
    >
    > Set `${ENCODER_INPUT_FEATURES_DTYPE}` to `TYPE_BF16` for mllama, and `TYPE_FP16` for other models.
    > `cross_kv_cache_fraction` is used to determine the paged kv cache memory pool size of enc-dec models. For such case, we distinguish `free_fraction * (1 - cross_kv_cache_fraction)` to self attention kv caches, and `free_fraction * cross_kv_cache_fraction` to cross attention kv caches.

4. Launch Tritonserver

    ```bash
    python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 1 --model_repo=multimodal_ifb/ --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    ```

    > **NOTE**:
    > If there is an error associated with 'MPI_Init_thread', please do `export PMIX_MCA_gds=hash`'
    >
    > When launching the server, since the prompt_embedding_table is in GPU memory, we need to set the CUDA pool memory for inter-step communication. For example, when we have a shape of (1, 576, 4096) promp_embedding table, we would need 300MB of CUDA pool memory, so we set 30MB to have some GPU buffers. (2(fp16=>2bytes) * 576 * 4096 * 8(max_batch_size) = 18,874,368)
    >
    > Also, the tensorrt_llm initialization assumes using another GPU, we need to initialize it but not use them.

### Send requests
1. Send request with `decoupled_mode` set to False
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 41.942 ms
    ```
2. Send request with `decoupled_mode` set to True
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --streaming

    [beam 0 ]:   sing
    [beam 0 ]:  apore
    [beam 0 ]:
    [INFO] Latency: 43.441 ms
    ```
3. Send request to the `tensorrt_llm_bls` model
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --use_bls

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 44.152 ms
    ```

4. Send request to the `tensorrt_llm_bls` model with `accumulate_tokens` set to True
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --use_bls --streaming

    [beam 0 ]:   sing
    [beam 0 ]:   singapore
    [beam 0 ]:   singapore
    [INFO] Latency: 45.48 ms
    ```

5. Send request with `enable_kv_cache_reuse` set to True
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text 'Question: which city is this? Answer:' --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 16 --model_type blip2 --prompt_table_extra_id ${id}

    [beam 0 ]:
    Question: which city is this? Answer: singapore
    [INFO] Latency: 42.514 ms
    ```
6. Send request with multiple images per request
    ```bash
    wget -O av.png https://raw.githubusercontent.com/Efficient-Large-Model/VILA/main/demo_images/av.png

    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text '<image>\n<image>\n Please elaborate what you see in the images?' --image av.png,'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' --request-output-len 68 --model_type vila --hf_model_dir ${HF_MODEL_PATH}

    [beam 0 ]:
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:  \n \n Please elaborate what you see in the images? ASSISTANT: The first image shows a busy street scene with a car driving through a crosswalk, surrounded by pedestrians and traffic lights. The second image captures a beautiful sunset with the iconic Merlion statue spouting water into the bay, with the Singapore Flyer and the city skyline in the background.

    [INFO] Latency: 403.879 ms
    ```

7. Send request with curl
    The triton server supports curl requests with an image url in the payload. For example here is a request sent to a Llama-3.2-11B-Vision (mLLama) model:
    ``` bash
    curl -X POST localhost:8000/v2/models/ensemble/generate_stream \
    -d '{"id": "42", "text_input": "<|image|>If I had to write a haiku for this one", "image_url_input": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png", "parameters": {"max_tokens": 16, "beam_width": 1, "end_id": 128001, "pad_id": 128004, "top_k": 1, "top_p": 0, "stream": false, "temperature": 0}}'

    # response
    data: {"batch_index":0,"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"id":"42","model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_index":0,"sequence_start":false,"text_output":"If I had to write a haiku for this one, it would be:.\\nMerlion spouts water.\\nMarina"}
   ```
   You can also send requests with base64 encoded images. Just replace the url above with `data:image/jpeg;base64,<base64_encoded_image>`.

8. Send request with video input
    ```bash
    python tensorrt_llm/triton_backend/tools/multimodal/client.py --text "Why is this video funny?" --video sample_demo_1.mp4 --video_num_frames 8 --request-output-len 30 --model_type llava_onevision  --end-id 151645

    [beam 0 ]:
    user
    Why is this video funny?assistant
    The video is funny because the child's actions are playful and exaggerated, as if they are reading the book with great enthusiasm.
    [INFO] Latency: 507.537 ms
    ```

> **NOTE**:
> Please ignore any exception thrown with the output. It's a known issue to be fixed.
>
> When `enable_kv_cache_reuse` is set to true, the `prompt_table_extra_id` must be specified in the requests. The `prompt_table_extra_id` is a unique identifier representing the image (or prompt table), the same image uses the same id. The data type is `uint64`, and the minimum value is 1.

### Kill the server
```bash
pkill tritonserver
```

### Supported image input types
When programmatically preparing your own request for the server, note that `ensemble`:
- `image_input`: a float16 5D tensor of shape `[batch_size, num_images, num_channels, height, width]` or `[batch_size, num_images, height, width, num_channels]` representing a batch of images already processed (via transformers AutoProcessor) for the vision encoder.
- `image_bytes_input`: a uint8 5D tensor of shape `[batch_size, num_images, num_channels, height, width]` or `[batch_size, num_images, height, width, num_channels]` representing a batch of raw images.
- `image_url_input`: a list of strings of shape `[batch_size, num_images]` representing a batch of image urls.

You may populate only one of these image inputs in a request. We suggest you use `image_bytes_input` when using grpc requests and `image_url_input` when sending http requests. For grpc requests where the client can preprocess images to reduce load on the server, use `image_input`. Note that `tensorrt_llm_bls` only supports `image_input`.

### Long multimodal context, FP8 KV cache and tensor parallelism

Follow these steps to enable chunked context inference (using LLaVA as an example) with FP8 KV cache and 2-way tensor parallelism. Ensure you convert the checkpoint using `--tp_size 2` and build the model with `--use_paged_context_fmha enable` and `--use_fp8_context_fmha enable`. Set the chunked context to true in the Tritonserver configuration file. The chunk size is determined by the `max_num_tokens` flag when building the engine, which defaults to 8192. When launching the server, you need to change the `--world_size` to match your tensor parallelism size.
1. Build the engine
```bash
    export MODEL_NAME="llava-1.5-7b-hf"
    export HF_MODEL_PATH=tmp/hf_models/${MODEL_NAME}

    # Convert checkpoint
    # For fp16 KV cache
    export UNIFIED_CKPT_PATH=tmp/trt_models/${MODEL_NAME}/fp8/2-gpu
    export ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/fp8/2-gpu
    export MULTIMODAL_ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/multimodal_encoder
    python tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir ${HF_MODEL_PATH} \
        --output_dir ${UNIFIED_CKPT_PATH} \
        --dtype float16 \
        --tp_size 2

    # For fp8 KV cache
    export UNIFIED_CKPT_PATH=tmp/trt_models/${MODEL_NAME}/fp8/2-gpu
    export ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/fp8/2-gpu
    export MULTIMODAL_ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/multimodal_encoder
    python ./tensorrt_llm/examples/quantization/quantize.py \
                                --model_dir ${HF_MODEL_PATH} \
                                --dtype float16 \
                                --qformat fp8 \
                                --kv_cache_dtype fp8 \
                                --output_dir ${UNIFIED_CKPT_PATH} \
                                --calib_size 512 \
                                --tp_size 2

    # Build the llm engine
    # --use_paged_context_fmha and --use_fp8_context_fmha are defaultly enabled
    # include --max_num_tokens to set the chunk size
    trtllm-build \
        --checkpoint_dir ${UNIFIED_CKPT_PATH} \
        --output_dir ${ENGINE_PATH} \
        --gemm_plugin auto \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 4608 # 8 (max_batch_size) * 576 (num_multimodal_features) for LLaVA

    # Build the multimodal engine
    python tensorrt_llm/examples/multimodal/build_multimodal_engine.py --model_path ${HF_MODEL_PATH} --model_type llava --max_batch_size 8 --output_dir ${MULTIMODAL_ENGINE_PATH}
```
2. Prepare the Tritonserver config file
Prepare the Tritonserver config file with `enable_chunked_context` set to True. Also, to further utilize the free memory, we can set `kv_cache_free_gpu_mem_fraction` to 0.9.
```bash
cp tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/ multimodal_ifb -r
# Override the ensemble and creates new multimodal_encoders directories for multimodal
cp tensorrt_llm/triton_backend/all_models/multimodal/ensemble multimodal_ifb -r
cp tensorrt_llm/triton_backend/all_models/multimodal/multimodal_encoders multimodal_ifb -r

# Changes the enable_chunked_context to True, and set kv_cache_free_gpu_mem_fraction to 0.9
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:True,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,kv_cache_free_gpu_mem_fraction:0.9

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1,max_queue_delay_microseconds:20000

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/ensemble/config.pbtxt triton_max_batch_size:8,logits_datatype:TYPE_FP32

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

# Newly added for multimodal
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i multimodal_ifb/multimodal_encoders/config.pbtxt triton_max_batch_size:8,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH},max_queue_delay_microseconds:20000
```
3. Launch the server
```bash
# Change --world_size to your tp size
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 2 --model_repo=multimodal_ifb/ --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders --multimodal_gpu0_cuda_mem_pool_bytes 300000000
```

When you launch the server, you will see logs similar to the following. In theory, now you can process long multimodal context up to the "max tokens in paged KV cache" value, and the context prefill phase will be done in chunk sizes.
```bash
[TensorRT-LLM][INFO] Memory usage when calculating max tokens in paged kv cache: total: 93.10 GiB, available: 85.57 GiB
...
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 77.02 GiB for max tokens in paged KV cache (315488).
```
