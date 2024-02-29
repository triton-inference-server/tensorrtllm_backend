#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
BS=$3
MAX_INPUT_SEQLEN=$4
MAX_OUTPUT_SEQLEN=$5
MAX_TOKENS=$6
TP=$7
PP=$8
WORLD_SIZE=$9

GPT2=/trt_llm_data/llm-models/gpt2
OPT_125M=/trt_llm_data/llm-models/opt-125m
LLAMA=/trt_llm_data/llm-models/llama-models/llama-7b-hf
GPTJ=/trt_llm_data/llm-models/gpt-j-6b
MISTRAL=/trt_llm_data/llm-models/mistral-7b-v0.1

set -e
pushd ../../

if [ "$MODEL" = "llama-7b-fp16" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 32 --n_head 32 --n_embd 4096 --inter_size 11008 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --use_gemm_plugin float16 \

    popd

fi

if [ "$MODEL" = "mistral-7b-fp16" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --model_dir /tensorrtllm_backens/models/Mistral-7B-v0.1  --dtype float16 \
      --use_gpt_attention_plugin float16  \
      --use_gemm_plugin float16  \
      --output_dir "$ENGINE_PATH"  \
      --max_batch_size "$BS" --max_input_len 32256 --max_output_len 512 \
      --enable_context_fmha --remove_input_padding \
      --use_inflight_batching --paged_kv_cache \
      --max_num_tokens "$MAX_TOKENS"

    popd

fi

if [ "$MODEL" = "llama-7b-fp8" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 32 --n_head 32 --n_embd 4096 --inter_size 11008 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --enable_fp8 \
        --fp8_kv_cache \
        --strongly_typed

    popd

fi

if [ "$MODEL" = "llama-13b-fp8" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 40 --n_head 40 --n_embd 5120 --inter_size 13824 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --enable_fp8 \
        --fp8_kv_cache \
        --strongly_typed

    popd

fi

if [ "$MODEL" = "llama-13b-fp16" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 40 --n_head 40 --n_embd 5120 --inter_size 13824 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --use_gemm_plugin float16

    popd

fi

if [ "$MODEL" = "llama-70b-fp8" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    if [ "$PP" > 1 ]; then
        # Use gen_micro_batch_size as max_batch_size for engine build
        ENGINE_BS=$(expr $BS / $PP)
    else
        ENGINE_BS=$BS
    fi

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$ENGINE_BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 80 --n_head 64 --n_kv_head 8 --n_embd 8192 --inter_size 28672 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --ffn_dim_multiplier 1.3 --multiple_of 4096 \
        --enable_fp8 \
        --fp8_kv_cache \
        --strongly_typed

    popd

fi

if [ "$MODEL" = "llama-70b-fp16" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    if [ "$PP" > 1 ]; then
        # Use gen_micro_batch_size as max_batch_size for engine build
        ENGINE_BS=$(expr $BS / $PP)
    else
        ENGINE_BS=$BS
    fi

    python3 build.py --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$ENGINE_BS" \
        --max_input_len "$MAX_INPUT_SEQLEN" \
        --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --use_inflight_batching \
        --paged_kv_cache \
        --max_num_tokens "$MAX_TOKENS" \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --n_layer 80 --n_head 64 --n_kv_head 8 --n_embd 8192 --inter_size 28672 \
        --vocab_size 32000 --n_positions 4096 --hidden_act "silu" \
        --ffn_dim_multiplier 1.3 --multiple_of 4096 \
        --use_gemm_plugin float16

    popd

fi

if [ "$MODEL" = "gptj-6b-fp8" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt

    # No pipeline parallelism argument in build.py for now.
    python3 build.py --dtype=float16 \
        --use_gpt_attention_plugin float16 \
        --max_batch_size "$BS" --max_input_len "$MAX_INPUT_SEQLEN" --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --vocab_size 50401 --max_beam_width 1 \
        --output_dir "$ENGINE_PATH" \
        --model_dir /mlperf_inference_data/models/GPTJ-6B/checkpoint-final \
        --enable_context_fmha \
        --fp8_kv_cache \
        --enable_fp8 \
        --parallel_build \
        --world_size "$WORLD_SIZE" \
        --paged_kv_cache \
        --use_inflight_batching \
        --remove_input_padding \
        --strongly_typed \
        --max_num_tokens "$MAX_TOKENS"

    popd

fi

if [ "$MODEL" = "gptj-6b-fp16" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt

    # No pipeline parallelism argument in build.py for now.
    python3 build.py --dtype=float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --max_batch_size "$BS" --max_input_len "$MAX_INPUT_SEQLEN" --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --vocab_size 50401 --max_beam_width 1 \
        --output_dir "$ENGINE_PATH" \
        --model_dir /mlperf_inference_data/models/GPTJ-6B/checkpoint-final \
        --enable_context_fmha \
        --paged_kv_cache \
        --parallel_build \
        --world_size "$WORLD_SIZE" \
        --use_inflight_batching \
        --remove_input_padding \
        --max_num_tokens "$MAX_TOKENS"

    popd

fi

if [ "$MODEL" = "falcon-180b-fp8" ]; then

    pushd tensorrt_llm/examples/falcon

    pip install -r requirements.txt

    python3 build.py --use_inflight_batching \
        --paged_kv_cache \
        --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype bfloat16 \
        --use_gpt_attention_plugin bfloat16 \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --max_batch_size "$BS" --max_input_len "$MAX_INPUT_SEQLEN" --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --enable_fp8 --fp8_kv_cache \
        --strongly_typed \
        --n_layer 80 --n_head 232 --n_kv_head 8 --n_embd 14848 --vocab_size 65024 --new_decoder_architecture \
        --max_num_tokens "$MAX_TOKENS"

    popd

fi

if [ "$MODEL" = "falcon-180b-fp16" ]; then

    pushd tensorrt_llm/examples/falcon

    pip install -r requirements.txt

    python3 build.py --use_inflight_batching \
        --paged_kv_cache \
        --remove_input_padding \
        --enable_context_fmha \
        --parallel_build \
        --output_dir "$ENGINE_PATH" \
        --dtype bfloat16 \
        --use_gemm_plugin bfloat16 \
        --use_gpt_attention_plugin bfloat16 \
        --world_size "$WORLD_SIZE" \
        --tp_size "$TP" \
        --pp_size "$PP" \
        --max_batch_size "$BS" --max_input_len "$MAX_INPUT_SEQLEN" --max_output_len "$MAX_OUTPUT_SEQLEN" \
        --n_layer 80 --n_head 232 --n_kv_head 8 --n_embd 14848 --vocab_size 65024 --new_decoder_architecture \
        --max_num_tokens "$MAX_TOKENS"

    popd

fi
