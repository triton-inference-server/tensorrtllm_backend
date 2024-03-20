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

pkill -9 -f tritonserver || true

# install deps
pip3 install -r tensorrt_llm/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

if [ "$MODEL" = "gpt" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16 | remove_input_padding"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --max_batch_size 8 --max_input_len 924 --max_output_len 100 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/

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
                --gemm_plugin float16 \
                --gpt_attention_plugin float16 \
                --context_fmha=enable \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_output_len 100 \
                --output_dir trt_engine/opt-125m/fp16/1-gpu/


    popd # tensorrt_llm/examples/opt

fi

if [ "$MODEL" = "llama" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    echo "Convert LLaMA from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/llama-7b/fp16

    echo "Build LLaMA"
    trtllm-build --model_config ./c-model/llama-7b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --output_dir llama_outputs

    python3 ../run.py --max_output_len=1 --engine_dir llama_outputs --tokenizer_dir=${LLAMA}

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "mistral" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    echo "Convert Mistral from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/mistral-7b/fp16

    echo "Build Mistral"
    trtllm-build --model_config ./c-model/mistral-7b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_input_len 8192 \
        --max_batch_size 8 \
        --output_dir mistral_7b_outputs

    # Equivalent to LLaMA at this stage except the tokenizer
    python3 ../run.py --max_output_len=1 --tokenizer_dir=${MISTRAL} --max_attention_window_size=4096 --engine_dir mistral_7b_outputs

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "mistral-ib" ]; then

    pushd tensorrt_llm/examples/llama

    pip install -r requirements.txt

    echo "Convert Mistral from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/mistral-7b/fp16

    echo "Build Mistral with inflight batching"
    trtllm-build --model_config ./c-model/mistral-7b/fp16/config.json  \
        --context_fmha=enable \
        --remove_input_padding=enable \
        --paged_kv_cache=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_input_len 8192 \
        --output_dir ib_mistral_7b_outputs

    popd # tensorrt_llm/examples/llama

fi

if [ "$MODEL" = "gptj" ]; then

    pushd tensorrt_llm/examples/gptj

    pip install -r requirements.txt

    echo "Convert GPT-J from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/gpt-j-6b/fp16

    echo "Build GPT-J"
    trtllm-build --model_config ./c-model/gpt-j-6b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --output_dir gptj_outputs

    python3 ../run.py --max_output_len=1 --tokenizer_dir=${GPTJ} --engine_dir gptj_outputs

    popd # tensorrt_llm/examples/gptj

fi

if [ "$MODEL" = "gpt-ib" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --context_fmha_fp32_acc enable \
        --use_paged_context_fmha enable \
        --gemm_plugin float16 \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt2-ib/fp16/1-gpu/

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-medium-ib" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2_MEDIUM} --dtype float16 --output_dir ./c-model/gpt2-medium/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2-medium/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --context_fmha enable \
        --use_paged_context_fmha enable \
        --max_draft_len 5 \
        --max_batch_size 8 --max_input_len 924 --max_output_len 128 \
        --output_dir trt_engine/gpt2-medium-ib/fp16/1-gpu/

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from NeMo"
    python3 convert_checkpoint.py --nemo_ckpt_path ${GPT2_NEXT_PTUNING}/megatron_converted_8b_tp4_pp1.nemo --dtype float16 --output_dir ./c-model/email_composition/fp16

    echo "Convert ptuning table"
    python3 nemo_prompt_convert.py -i ${GPT2_NEXT_PTUNING}/email_composition.nemo -o email_composition.npy

    cp ${GPT2_NEXT_PTUNING}/input.csv ./

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/email_composition/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --context_fmha enable \
        --max_batch_size 4 --max_input_len 128 --max_output_len 128 --max_beam_width 1 \
        --output_dir trt_engine/email_composition/fp16/1-gpu/ \
        --max_prompt_embedding_table_size 300

    popd # tensorrt_llm/examples/gpt

fi

if [ "$MODEL" = "gpt-2b-ib-lora" ]; then

    # GPT-2B
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from NeMo"
    python3 convert_checkpoint.py --nemo_ckpt_path ${GPT_2B} --dtype float16 --output_dir ./c-model/gpt-2b-lora/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt-2b-lora/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --lora_plugin float16 \
        --lora_dir ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        --lora_ckpt_source nemo \
        --lora_target_modules attn_qkv \
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

if [ "$MODEL" = "gpt-gather-logits" ]; then

    # GPT2
    pushd tensorrt_llm/examples/gpt

    pip3 install -r requirements.txt

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16 | gather_all_token_logits"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --context_fmha enable \
        --max_batch_size 128 --max_input_len 300 --max_output_len 300 \
        --gather_all_token_logits \
        --output_dir trt_engine/gpt2-gather-logits/fp16/1-gpu/ \
        --max_num_tokens 38400

    popd # tensorrt_llm/examples/gpt

fi
