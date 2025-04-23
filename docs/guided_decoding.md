# End-to-End Workflow for Guided Decoding with TensorRT-LLM Backend

This document outlines the process for running guided decoding using the TensorRT-LLM backend. Guided decoding ensures that generated outputs adhere to specified formats, such as JSON. Currently, this feature is supported through the [XGrammar](https://github.com/mlc-ai/xgrammar) backend.

For more information, refer to the [guided decoding documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/advanced/executor.md#structured-output-with-guided-decoding) from TensorRT-LLM. Additionally, you can explore another example of [guided decoding + LLM API example](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_guided_decoding.html).

## Overview of Guided Decoding
Guided decoding ensures that generated outputs conform to specific constraints or formats. Supported guide types include:
- **None**: No constraints.
- **JSON**: Outputs in JSON format.
- **JSON Schema**: JSON format with schema validation.
- **Regex**: Outputs matching a regular expression.
- **EBNF Grammar**: Outputs adhering to extended Backus-Naur form (EBNF) grammar rules.

# Build TensorRT-LLM engine and launch Tritonserver

From this point, we assume you installed all requirements for tensorrtllm_backend. You can refer to [build.md](build.md) for installation and docker launch.

## Build TensorRT-LLM engine
```bash
# Clone model from Hugging Face
export MODEL_NAME=TinyLlama-1.1B-Chat-v1.0
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 hf_models/${MODEL_NAME}

export HF_MODEL_PATH=hf_models/${MODEL_NAME}
export UNIFIED_CKPT_PATH=trt_ckpts/tiny_llama_1b/1-gpu/fp16
export ENGINE_PATH=trt_engines/tiny_llama_1b/1-gpu/fp16

python tensorrt_llm/examples/models/core/llama/convert_checkpoint.py --model_dir ${HF_MODEL_PATH} \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --kv_cache_type paged \
             --max_batch_size 64
```
## Launch Tritonserver

## Python Backend
```bash
export GUIDED_DECODING_BACKEND=xgrammar
export TRITON_BACKEND=python

cp all_models/inflight_batcher_llm/ llama_ifb -r

python3 tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,tokenizer_dir:${HF_MODEL_PATH},guided_decoding_backend:${GUIDED_DECODING_BACKEND}

python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/
```

## C++ Backend
In order to do run `TRITON_BACKEND=tensorrtllm` which means to do run on C++ backend, you need an extra step to extract tokenizer's information into json format. `generate_xgrammar_tokenizer_info.py` will create `xgrammar_tokenizer_info.json` under given output_dir argument. And we fill the `xgrammer_tokenizer_info_path` parameter in `tensorrt_llm/config.pbtxt`.
```bash
export XGRAMMAR_TOKENIZER_INFO_DIR=tokenizer_info/${MODEL_NAME}

python3 tensorrt_llm/examples/generate_xgrammar_tokenizer_info.py --model_dir ${HF_MODEL_PATH} --output_dir ${XGRAMMAR_TOKENIZER_INFO_DIR}

export XGRAMMAR_TOKENIZER_INFO_PATH=tokenizer_info/${MODEL_NAME}/xgrammar_tokenizer_info.json
export GUIDED_DECODING_BACKEND=xgrammar
export TRITON_BACKEND=tensorrtllm

cp all_models/inflight_batcher_llm/ llama_ifb -r

python3 tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,guided_decoding_backend:${GUIDED_DECODING_BACKEND},xgrammar_tokenizer_info_path:${XGRAMMAR_TOKENIZER_INFO_PATH}

python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/
```
# Sending Guided Decoding Requests

Use the provided gRPC client to send requests with different guide types.
```bash
# Set the prompt
PROMPT="What is the year after 2024? Answer:"

# 0. Guide type: None
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -p "${PROMPT}" -o 30 --exclude-input-in-output --verbose --model-name ensemble

# Output:
# 0: 2025
#
# Question 3: What is the year after 2025? Answer: 2026
#

# 1. Guide type: json
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -p  "${PROMPT}" -o 30 --exclude-input-in-output --verbose --model-name ensemble --guided-decoding-guide-type json

# Output:
# 0: [2025]

# 2. Guide type: json_schema
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -p  "${PROMPT}" -o 30 --exclude-input-in-output --verbose --model-name ensemble --guided-decoding-guide-type json_schema --guided-decoding-guide '{"properties": {"answer": {"title": "Answer", "type": "integer"}}, "required": ["answer"], "title": "Answer", "type": "object"}'

# Output:
# 0: {"answer": 2026}

# 3. Guide type: regex
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -p "${PROMPT}" -o 30 --exclude-input-in-output --verbose --model-name ensemble --guided-decoding-guide-type regex --guided-decoding-guide '\d+'

# Output:
# 0: 2025

# 4. Guide type: ebnf_grammar
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -p "${PROMPT}" -o 30 --exclude-input-in-output --verbose --model-name ensemble --guided-decoding-guide-type ebnf_grammar --guided-decoding-guide 'root ::= [0-9]+'

# Output:
# 0: 2025
```

Use curl method to send requests
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is the year after 2024? Answer:", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2, "guided_decoding_guide_type":"json"}'

# Output:
# {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"[2025]"}
```
