#!/usr/bin/bash

MODEL=$1

GPT2=/home/scratch.trt_llm_data/llm-models/gpt2
GPT2_MEDIUM=/home/scratch.trt_llm_data/llm-models/gpt2-medium
GPT2_NEXT_PTUNING=/home/scratch.trt_llm_data/llm-models/email_composition
OPT_125M=/home/scratch.trt_llm_data/llm-models/opt-125m
LLAMA=/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf
GPTJ=/home/scratch.trt_llm_data/llm-models/gpt-j-6b
MISTRAL=/home/scratch.trt_llm_data/llm-models/mistral-7b-v0.1
GPT_2B=/home/scratch.trt_llm_data/llm-models/GPT-2B-001_bf16_tp1.nemo
GPT_2B_LORA=/home/scratch.trt_llm_data/llm-models/lora/gpt-next-2b

set -e

# install deps
pip3 install -r tensorrt_llm/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

if [ "$MODEL" = "gpt" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i ${GPT2} -o ./c-model/gpt2/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT | remove_input_padding"
    python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --enable_context_fmha \
        --remove_input_padding \
        --max_batch_size 8 --max_input_len 924 --max_output_len 100 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/ --hidden_act gelu

    python3 ../run.py --max_output_len 10 --engine_dir=trt_engine/gpt2/fp16/1-gpu/ --tokenizer_dir ${GPT2}

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "opt" ]; then

    pushd tensorrt_llm/examples/opt

    pip install -r requirements.txt

    echo "Convert OPT from HF"
    python3 convert_checkpoint.py --model_dir ${OPT_125M} --dtype float16 --output_dir ./c-model/opt-125m/fp16

    echo "OPT builder"
    trtllm-build --checkpoint_dir ./c-model/opt-125m/fp16  \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_output_len 100 \
                --output_dir trt_engine/opt-125m/fp16/1-gpu/


    popd # tensorrt_llm/examples/opt

fi

if [ "$MODEL" = "llama" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for LLaMA
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin \
        --output_dir llama_outputs
    python3 ../run.py --max_output_len=1 --engine_dir llama_outputs --tokenizer_dir=${LLAMA}

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "mistral" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for Mistral
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin \
        --output_dir mistral_7b_outputs --max_input_len=8192
    # Equivalent to LLaMA at this stage except the tokenizer
    python3 ../run.py --max_output_len=1 --tokenizer_dir=${MISTRAL} --max_attention_window_size=4096 --engine_dir mistral_7b_outputs

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "mistral-ib" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for Mistral
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha --use_inflight_batching --paged_kv_cache \
        --use_gpt_attention_plugin --use_gemm_plugin \
        --output_dir ib_mistral_7b_outputs --max_input_len=8192

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "gptj" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt
    # Dummy weights because 7B is the minimal size for GPT-J
    python3 build.py --dtype=float16 --n_layer=2 \
        --enable_context_fmha \
        --use_gpt_attention_plugin --use_gemm_plugin
    python3 ../run.py --max_output_len=1 --tokenizer_dir=${GPTJ}

    popd # tensorrt_llm/examples/gptj

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
        --remove_input_padding \
        --paged_kv_cache \
        --enable_context_fmha_fp32_acc \
        --use_paged_context_fmha \
        --use_gemm_plugin float16 \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt2-ib/fp16/1-gpu/ --hidden_act gelu

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-medium-ib" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i ${GPT2_MEDIUM} -o ./c-model/gpt2-medium/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/gpt2-medium/fp16/1-gpu \
        --dtype float16 \
        --use_inflight_batching \
        --use_gpt_attention_plugin float16 \
        --paged_kv_cache \
        --use_gemm_plugin float16 \
        --enable_context_fmha --use_paged_context_fmha \
        --remove_input_padding --max_draft_len 5 \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt2-medium-ib/fp16/1-gpu/ --hidden_act gelu

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
        --remove_input_padding \
        --max_batch_size 4 --max_input_len 128 --max_output_len 128 --max_beam_width 1 \
        --output_dir trt_engine/email_composition/fp16/1-gpu/ --hidden_act gelu --enable_context_fmha \
        --max_prompt_embedding_table_size 300

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-2b-ib-lora" ]; then

    # GPT-2B
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from NeMo"
    python3 nemo_ckpt_convert.py -i ${GPT_2B} -o ./c-model/gpt-2b-lora/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/gpt-2b-lora/fp16/1-gpu \
        --dtype float16 \
        --use_inflight_batching \
        --use_gpt_attention_plugin float16 \
        --paged_kv_cache \
        --use_gemm_plugin float16 \
        --use_lora_plugin float16 \
        --lora_target_modules attn_qkv \
        --remove_input_padding \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt-2b-lora-ib/fp16/1-gpu/

    python3 nemo_lora_convert.py -i ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        -o gpt-2b-lora-train-900 --write-cpp-runtime-tensors --storage-type float16
    python3 nemo_lora_convert.py -i ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        -o gpt-2b-lora-train-900-tllm --storage-type float16
    cp ${GPT_2B_LORA}/gpt2b_lora-900.nemo .

    cp ${GPT_2B_LORA}/input.csv .

    popd # tensorrt_llm/examples/gpt
fi
