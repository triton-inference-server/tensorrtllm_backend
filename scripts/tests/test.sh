#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4

set -e
nvidia-smi
source tools/utils.sh

if [ "$MODEL" = "gpt" ] || [ "$MODEL" = "opt" ] || [ "$MODEL" = "llama" ] || [ "$MODEL" = "gptj" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/gpt/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH}
    python3 tools/fill_template.py -i all_models/gpt/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i all_models/gpt/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

    # Launch Triton Server
    mpirun --allow-run-as-root \
        -n 1 /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/gpt \
        --disable-auto-complete-config \
        --backend-config=python,shm-region-prefix-name=prefix0_ : &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    pushd tools/gpt/

    # Client
    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=http \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

    # Async Client
    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=http \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

    # End to end test
    python3 end_to_end_test.py \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

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
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:False
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/inflight_batcher_llm &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py \
        --check-output \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}
    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm

    python3 end_to_end_test.py \
        --concurrency 8 \
        -i http \
        --max_input_len 300 \
        --dataset ../dataset/mini_cnn_eval.json
    python3 end_to_end_test.py \
        --concurrency 8 \
        -i grpc \
        --max_input_len 300 \
        --dataset ../dataset/mini_cnn_eval.json

    python3 identity_test.py \
        --concurrency 8 \
        -i http \
        --max_input_len 300 \
        --dataset ../dataset/mini_cnn_eval.json \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}
    python3 identity_test.py \
        --concurrency 8 \
        -i grpc \
        --max_input_len 300 \
        --dataset ../dataset/mini_cnn_eval.json \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}

    popd # tools/inflight_batcher_llm

    kill ${SERVER_PID}

fi

if [ "$MODEL" = "gpt-ib-streaming" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:True
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/inflight_batcher_llm &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py \
        --streaming --check-output \
        --tokenizer_dir ${TOKENIZER_PATH} \
        --tokenizer_type ${TOKENIZER_TYPE}
    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm
    python3 end_to_end_streaming_client.py \
        --output_len 10 --prompt "This is a test "
    popd

    kill ${SERVER_PID}

fi
