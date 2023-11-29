#!/usr/bin/bash

MODEL=$1

GPT2=/home/scratch.trt_llm_data/llm-models/gpt2
GPT2_NEXT_PTUNING=/home/scratch.trt_llm_data/llm-models/email_composition
OPT_125M=/home/scratch.trt_llm_data/llm-models/opt-125m
LLAMA=/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf
GPTJ=/home/scratch.trt_llm_data/llm-models/gpt-j-6b
MISTRAL=/home/scratch.trt_llm_data/llm-models/mistral-7b-v0.1

set -e

# install deps
pip3 install -r tensorrt_llm/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

if [ "$MODEL" = "gpt" ]; then

    # GPT2
    pushd tensorrt_llm/examples
    pushd gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i ${GPT2} -o ./c-model/gpt2/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT | remove_input_padding"
    python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_layernorm_plugin float16 \
        --enable_context_fmha \
        --remove_input_padding \
        --max_batch_size 8 --max_input_len 924 --max_output_len 100 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/ --hidden_act gelu

    popd # gpt

    python3 run.py --max_output_len 10 --engine_dir=gpt/trt_engine/gpt2/fp16/1-gpu/

    popd # tensorrt_llm/examples

fi

if [ "$MODEL" = "opt" ]; then

    pushd tensorrt_llm/examples/opt

    pip install -r requirements.txt

    echo "Convert OPT from HF"
    python3 hf_opt_convert.py -i ${OPT_125M} -o ./c-model/opt-125m/fp16 -i_g 1 -weight_data_type fp16

    echo "OPT builder"
    python3 build.py --model_dir=./c-model/opt-125m/fp16/1-gpu/ \
                     --max_batch_size 8 \
                     --use_gpt_attention_plugin float16 \
                     --use_gemm_plugin float16 \
                     --use_layernorm_plugin float16 \
                     --enable_context_fmha \
                     --max_input_len 924 \
                     --max_output_len 100 \
                     --world_size 1 \
                     --output_dir trt_engine/opt-125m/fp16/1-gpu/ \
                     --do_layer_norm_before \
                     --pre_norm \
                     --hidden_act relu

    popd # tensorrt_llm/examples/opt

fi

if [ "$MODEL" = "llama" ]; then

    pushd tensorrt_llm/examples
    pushd llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for LLaMA
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin --use_rmsnorm_plugin --output_dir llama_outputs
    popd # llama
    python3 run.py --max_output_len=1 --tokenizer_dir=${LLAMA} --engine_dir=llama/llama_outputs

    popd # tensorrt_llm/examples

fi

if [ "$MODEL" = "mistral" ]; then

    pushd tensorrt_llm/examples
    pushd llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for Mistral
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin --use_rmsnorm_plugin \
        --output_dir mistral_7b_outputs --max_input_len=8192
    popd # llama
    # Equivalent to LLaMA at this stage except the tokenizer
    python3 run.py --max_output_len=1 --tokenizer_dir=${MISTRAL} --max_kv_cache_len=4096 --engine_dir llama/mistral_7b_outputs

    popd # tensorrt_llm/examples

fi

if [ "$MODEL" = "mistral-ib" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for Mistral
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha --use_inflight_batching --paged_kv_cache \
        --use_gpt_attention_plugin --use_gemm_plugin --use_rmsnorm_plugin \
        --output_dir ib_mistral_7b_outputs --max_input_len=8192

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "gptj" ]; then

    pushd tensorrt_llm/examples
    pushd gptj

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for GPT-J
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin
    popd  # gptj
    python3 run.py --max_output_len=1 --engine_dir=gptj/engine_outputs

    popd # tensorrt_llm/examples

fi

if [ "$MODEL" = "gpt-ib" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i ${GPT2} -o ./c-model/gpt2/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
        --dtype float16 \
        --use_inflight_batching \
        --use_gpt_attention_plugin float16 \
        --paged_kv_cache \
        --use_gemm_plugin float16 \
        --use_layernorm_plugin float16 \
        --remove_input_padding \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt2-ib/fp16/1-gpu/ --hidden_act gelu

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 nemo_ckpt_convert.py -i ${GPT2_NEXT_PTUNING}/megatron_converted_8b_tp4_pp1.nemo -o ./c-model/email_composition/fp16 --storage-type float16 --tensor-parallelism 1 --processes 1
#
    echo "Convert ptuning table"
    python3 nemo_prompt_convert.py -i ${GPT2_NEXT_PTUNING}/email_composition.nemo -o email_composition.npy

    cp ${GPT2_NEXT_PTUNING}/input.csv ./

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/email_composition/fp16/1-gpu \
        --use_inflight_batching \
        --use_gpt_attention_plugin \
        --paged_kv_cache \
        --use_gemm_plugin \
        --use_layernorm_plugin \
        --remove_input_padding \
        --max_batch_size 4 --max_input_len 128 --max_output_len 128 --max_beam_width 1 \
        --output_dir trt_engine/email_composition/fp16/1-gpu/ --hidden_act gelu --enable_context_fmha \
        --max_prompt_embedding_table_size 300

    popd # tensorrt_llm/examples/gpt

fi
