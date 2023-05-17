#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2

set -e
nvidia-smi
source tools/utils.sh

if [ "$MODEL" = "GPT" ]; then
    # Modify config.pbtxt
    bash tools/gpt/create_gpt_config.sh ${ENGINE_PATH}
    mv config.pbtxt all_models/gpt/tekit

    # Launch Triton Server
    mpirun --allow-run-as-root \
        -n 1 /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/gpt \
        --backend-config=python,shm-region-prefix-name=prefix0_ : &
    export SERVER_PID=$!
    wait_for_server_ready $SERVER_PID 1200

    pushd examples/gpt
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

    # Client
    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10

    popd # examples/gpt

    pushd tools/gpt

    # Identity test
    python3 identity_test.py \
        --batch_size=8 --start_len=128 --output_len=20

    # Benchmark using Perf Analyzer
    python3 gen_input_data.py
    perf_analyzer -m tekit \
        -b 8 --input-data input_data.json \
        --concurrency-range 2 \
        -i http \
        -u 'localhost:8000'

    perf_analyzer -m tekit \
        -b 8 --input-data input_data.json \
        --concurrency-range 2 \
        -i grpc \
        -u 'localhost:8001'

    popd # tools/gpt

fi
