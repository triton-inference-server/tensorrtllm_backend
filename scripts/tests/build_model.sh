#!/usr/bin/bash

MODEL=$1

set -e

# install deps
pip3 install -r tekit/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

if [ "$MODEL" = "gpt" ]; then

    # GPT2
    pushd tekit/examples/gpt

    pip3 install -r requirements.txt
    pip3 uninstall -y safetensors # To fix `safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge`

    rm -rf gpt2 && git clone https://huggingface.co/gpt2
    pushd gpt2 && rm pytorch_model.bin && \
        wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2/fp16 --storage-type float16

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --use_layernorm_plugin float16 \
        --max_batch_size 8 --max_input_len 924 --max_output_len 100 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/ --hidden_act gelu

    python3 run.py --max_output_len 10 --engine_dir=trt_engine/gpt2/fp16/1-gpu/

    popd # tekit/examples/gpt

fi

if [ "$MODEL" = "opt" ]; then

    pushd tekit/examples/opt

    pip install -r requirements.txt

    mkdir opt-125m && pushd opt-125m && \
    wget -q https://huggingface.co/facebook/opt-125m/resolve/main/config.json && \
    wget https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin && \
    wget https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json && \
    wget https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer_config.json && \
    wget https://huggingface.co/facebook/opt-125m/resolve/main/generation_config.json && \
    wget https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt && popd

    echo "Convert OPT from HF"
    python3 hf_opt_convert.py -i opt-125m/ -o ./c-model/opt-125m/fp16 -i_g 1 -weight_data_type fp16

    echo "OPT builder"
    python3 build.py --model_dir=./c-model/opt-125m/fp16/1-gpu/ \
                     --max_batch_size 8 \
                     --use_gpt_attention_plugin float16 \
                     --use_gemm_plugin float16 \
                     --use_layernorm_plugin float16 \
                     --max_input_len 924 \
                     --max_output_len 100 \
                     --world_size 1 \
                     --output_dir trt_engine/opt-125m/fp16/1-gpu/ \
                     --do_layer_norm_before \
                     --pre_norm \
                     --hidden_act relu

    popd # tekit/examples/opt

fi

if [ "$MODEL" = "llama" ]; then

    pushd tekit/examples/llama

    pip install -r requirements.txt
    python3 build.py --dtype=float16 --n_layer=2 \
        --use_gpt_attention_plugin --use_gemm_plugin
    wget -q https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/tokenizer.model
    python3 run.py --max_output_len=1

    popd # tekit/examples/llama

fi

if [ "$MODEL" = "gptj" ]; then

    pushd tekit/examples/gptj

    pip install -r requirements.txt
    python3 build.py --dtype=float16 --n_layer=2 \
        --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin
    wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json
    wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt
    python3 run.py --max_output_len=1

    popd # tekit/examples/gptj

fi
