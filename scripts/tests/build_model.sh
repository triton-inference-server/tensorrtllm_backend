#!/usr/bin/bash

MODEL=$1

set -e

# install deps
pip3 install -r tekit/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

if [ "$MODEL" = "GPT" ]; then

    # GPT2
    pushd tekit/examples/gpt

    pip3 install -r requirements.txt

    rm -rf gpt2 && git clone https://huggingface.co/gpt2
    pushd gpt2 && rm pytorch_model.bin && \
        wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

    echo "Convert GPT from HF"
    python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2/fp16 --storage-type fp16

    echo "Build GPT: float16 | src FT"
    python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
        --max_batch_size 8 --max_input_len 924 --max_output_len 100 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/ --hidden_act gelu

    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    python3 run.py --max_output_len 10 --engine_dir=trt_engine/gpt2/fp16/1-gpu/

    popd # tekit/examples/gpt

fi
