#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2

set -e
nvidia-smi
source tools/utils.sh

if [ "$MODEL" = "gpt" ] || [ "$MODEL" = "opt" ] || [ "$MODEL" = "llama" ] || [ "$MODEL" = "gptj" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/gpt/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH}

    # Launch Triton Server
    mpirun --allow-run-as-root \
        -n 1 /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/gpt \
        --disable-auto-complete-config \
        --backend-config=python,shm-region-prefix-name=prefix0_ : &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    pushd tools/gpt/
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O gpt2-merges.txt

    # Client
    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=http

    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc

    # Async Client
    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=http

    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc

    # End to end test
    python3 end_to_end_test.py

    # Identity test
    python3 identity_test.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=http --mode=sync

    python3 identity_test.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=grpc --mode=sync

    python3 identity_test.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=http --mode=async

    python3 identity_test.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=grpc --mode=async

    # Benchmark using Perf Analyzer
    python3 gen_input_data.py
    # FIXME(kaiyu): Uncomment this when perf_analyzer is available.
    # perf_analyzer -m tensorrt_llm -v \
    #     -b 8 --input-data input_data.json \
    #     --concurrency-range 2 \
    #     -i http \
    #     -u 'localhost:8000'

    # perf_analyzer -m tensorrt_llm -v \
    #     -b 8 --input-data input_data.json \
    #     --concurrency-range 2 \
    #     -i grpc \
    #     -u 'localhost:8001'

    kill ${SERVER_PID}

    popd # tools/gpt

fi

if [ "$MODEL" = "gpt-ib" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH}

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/inflight_batcher_llm &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py
    popd # inflight_batcher_llm/client

    kill ${SERVER_PID}

fi
