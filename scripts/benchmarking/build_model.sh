#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
BS=$3

GPT2=/trt_llm_data/llm-models/gpt2
OPT_125M=/trt_llm_data/llm-models/opt-125m
LLAMA=/trt_llm_data/llm-models/llama-models/llama-7b-hf
GPTJ=/trt_llm_data/llm-models/gpt-j-6b

set -e

pushd ../../

if [ "$MODEL" = "llama-7b-fp16" ]; then


    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /llama-models/v2/7B  --dtype float16 \
      --use_gpt_attention_plugin float16  \
      --use_gemm_plugin float16  \
      --output_dir "$ENGINE_PATH"  \
      --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
      --use_rmsnorm_plugin float16  \
      --enable_context_fmha --remove_input_padding \
      --use_inflight_batching --paged_kv_cache

    popd

fi

if [ "$MODEL" = "llama-7b-fp8" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /llama-models/v2/7B  --dtype float16 \
      --use_gpt_attention_plugin float16  \
      --use_gemm_plugin float16  \
      --output_dir "$ENGINE_PATH"  \
      --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
      --use_rmsnorm_plugin float16  \
      --enable_context_fmha --remove_input_padding \
      --use_inflight_batching --paged_kv_cache \
      --enable_fp8 --fp8_kv_cache \
      --max_num_tokens 35000

    popd

fi

if [ "$MODEL" = "llama-13b-fp8" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /llama-models/v2/7B  --dtype float16 \
      --use_gpt_attention_plugin float16  \
      --use_gemm_plugin float16  \
      --output_dir "$ENGINE_PATH"  \
      --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
      --use_rmsnorm_plugin float16  \
      --enable_context_fmha --remove_input_padding \
      --use_inflight_batching --paged_kv_cache \
      --enable_fp8 --fp8_kv_cache \
      --strongly_typed --n_layer 40 --n_head 40 --n_embd 5120 --inter_size 13824 --vocab_size 32000 --n_positions 4096 --hidden_act silu \
      --max_num_tokens 35000

    popd

fi

if [ "$MODEL" = "llama-70b-fp8-tp2" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /trt_llm_data/llm-models/llama-models-v2/70B  --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --use_inflight_batching \
        --remove_input_padding \
        --enable_context_fmha \
        --enable_fp8 \
        --fp8_kv_cache \
        --paged_kv_cache \
        --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
        --output_dir "$ENGINE_PATH" \
        --parallel_build \
        --world_size 2 \
        --tp_size 2

    popd

fi

if [ "$MODEL" = "llama-70b-fp8-tp4" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /trt_llm_data/llm-models/llama-models-v2/70B  --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --use_inflight_batching \
        --remove_input_padding \
        --enable_context_fmha \
        --enable_fp8 \
        --fp8_kv_cache \
        --paged_kv_cache \
        --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
        --output_dir "$ENGINE_PATH" \
        --parallel_build \
        --world_size 4 \
        --tp_size 4


    popd

fi

if [ "$MODEL" = "llama-70b-fp16-tp4" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --meta_ckpt_dir /trt_llm_data/llm-models/llama-models-v2/70B  --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_rmsnorm_plugin float16 \
        --use_inflight_batching \
        --remove_input_padding \
        --enable_context_fmha \
        --paged_kv_cache \
        --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
        --output_dir "$ENGINE_PATH" \
        --parallel_build \
        --world_size 4 \
        --tp_size 4

    popd

fi

if [ "$MODEL" = "gptj-6b-fp8" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt

    python3 build.py  --dtype=float16 \
        --use_gpt_attention_plugin float16  \
        --use_gemm_plugin float16 \
        --max_batch_size "$BS" --max_input_len 1535 --max_output_len 512 \
        --vocab_size 50401 --max_beam_width 1 \
        --output_dir "$ENGINE_PATH" \
        --model_dir /mlperf_inference_data/models/GPTJ-6B/checkpoint-final \
        --enable_context_fmha \
        --fp8_kv_cache \
        --enable_fp8 \
        --paged_kv_cache \
        --use_inflight_batching \
        --remove_input_padding

    popd

fi

if [ "$MODEL" = "gptj-6b-fp16" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt

    python3 build.py  --dtype=float16 \
        --use_gpt_attention_plugin float16  \
        --use_gemm_plugin float16 \
        --max_batch_size "$BS" --max_input_len 1535 --max_output_len 512 \
        --vocab_size 50401 --max_beam_width 1 \
        --output_dir "$ENGINE_PATH" \
        --model_dir /mlperf_inference_data/models/GPTJ-6B/checkpoint-final \
        --enable_context_fmha  \
        --paged_kv_cache \
        --use_inflight_batching \
        --remove_input_padding

    popd

fi

if [ "$MODEL" = "falcon-180b-fp8-tp8" ]; then

    pushd tensorrt_llm/examples/falcon

    pip install -r requirements.txt

    python3 build.py --use_inflight_batching \
        --paged_kv_cache \
        --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype bfloat16  \
        --use_gemm_plugin bfloat16 \
        --use_gpt_attention_plugin bfloat16 \
        --world_size 8 \
        --tp_size 8 \
        --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
        --enable_fp8 --fp8_kv_cache \
         --n_layer 80 --n_head 232 --n_kv_head 8 --n_embd 14848 --vocab_size 65024 --new_decoder_architecture

    popd

fi

if [ "$MODEL" = "falcon-180b-fp16-tp8" ]; then

    pushd tensorrt_llm/examples/falcon

    pip install -r requirements.txt


    python3 build.py --use_inflight_batching \
        --paged_kv_cache \
        --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype bfloat16  \
        --use_gemm_plugin bfloat16 \
        --use_gpt_attention_plugin bfloat16 \
        --world_size 8 \
        --tp_size 8 \
        --max_batch_size "$BS" --max_input_len 2048 --max_output_len 512 \
         --n_layer 80 --n_head 232 --n_kv_head 8 --n_embd 14848 --vocab_size 65024 --new_decoder_architecture

    popd

fi
