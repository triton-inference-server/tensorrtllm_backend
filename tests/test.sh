#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4

set -ex
set -o pipefail
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

print_test_params () {

    echo "----------------------------------"
    echo " Test parameters:"
    echo "----------------------------------"
    echo "BATCHING_STRATEGY: ${BATCHING_STRATEGY}"
    echo "MAX_NUM_SEQUENCES: ${MAX_NUM_SEQUENCE}"
    echo "MAX_TOKENS_IN_KV_CACHE: ${MAX_TOKENS_IN_KV_CACHE}"
    echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
    echo "KV_CACHE_FREE_GPU_MEM_FRACTION: ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
    echo "ENABLE_TRT_OVERLAP: ${ENABLE_TRT_OVERLAP}"
    echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
    echo "TRITON_MAX_BATCH_SIZE: ${TRITON_MAX_BATCH_SIZE}"
    echo "MAX_QUEUE_DELAY_MICROSECONDS: ${MAX_QUEUE_DELAY_MICROSECONDS}"
    echo "MAX_BEAM_WIDTH: ${MAX_BEAM_WIDTH}"
    echo "run_all_tests: ${run_all_tests}"
    echo "----------------------------------"
}

fill_triton_repo () {

    python3 tools/fill_template.py -i triton_repo/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},max_num_sequences:${MAX_NUM_SEQUENCE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:${ENABLE_TRT_OVERLAP},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_batch_width:${MAX_BEAM_WIDTH}
    python3 tools/fill_template.py -i triton_repo/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
    python3 tools/fill_template.py -i triton_repo/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
    python3 tools/fill_template.py -i triton_repo/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}

}

run_cpp_backend_tests () {

    print_test_params

    # Because the runners are shared, the default value of 0.85 doesn't work, so skip
    # if max_tokens_in_kv_cache is also empty
    if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
        echo "Skipping..."
        continue
    fi

    if [[ "${BATCHING_STRATEGY}" == "v1" && "${BATCH_SCHEDULER_POLICY}" == "max_utilization" ]]; then
        echo "Skipping. V1 doesn't support max_utilization"
        continue
    fi

    rm -rf ./triton_repo
    cp -R all_models/inflight_batcher_llm triton_repo

    # Modify config.pbtxt
    DECOUPLED_MODE="False"
    fill_triton_repo

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=triton_repo &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    # test to run for all combinations of flags
    EXCL_INPUT_IN_OUTPUT_FLAG=""
    [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"

    # Test client
    pushd inflight_batcher_llm/client

    python3 inflight_batcher_llm_client.py \
        --check-output \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}

    if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then

        test_stop_words ()
        {
            PROMPT="The only thing we have to fear is"
            OUTLEN=10

            ORIGINAL_OUTPUT=$(python3 end_to_end_grpc_client.py -o ${OUTLEN} -p "${PROMPT}" 2>&1)
            # should be something like "[...] that the government will [...]"

            # examples of stop words that won't affect generation
            # "government" isn't tokenized like " government"
            # " that the public" doesn't match entirely the generated string
            TEST_OUTPUT=$(python3 end_to_end_grpc_client.py -o ${OUTLEN} -p "${PROMPT}" --stop-words "government" " that the public" 2>&1)
            [[ "${ORIGINAL_OUTPUT}" == "${TEST_OUTPUT}" ]]

            # check that output finishes at "government"
            TEST_OUTPUT=$(python3 end_to_end_grpc_client.py -o ${OUTLEN} -p "${PROMPT}" --stop-words " lorem" " government" 2>&1)
            [[ "${TEST_OUTPUT}" == *"government']" ]]
            TEST_OUTPUT=$(python3 end_to_end_grpc_client.py -o ${OUTLEN} -p "${PROMPT}" --stop-words " that the government" 2>&1)
            [[ "${TEST_OUTPUT}" == *"government']" ]]
        }
        test_stop_words

        # test with embedding bias
        python3 end_to_end_grpc_client.py \
            -o 10 \
            -p "The only thing we have to fear is"  \
            --embedding-bias-words " government" \
            --embedding-bias-weights -20 \
            2>&1 | tee output_w_bias
        grep -v "that the government will" output_w_bias

        # test with request cancellation
        python3 inflight_batcher_llm_client.py \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            2>&1 | tee output_w_stop
        grep "Got cancellation response" output_w_stop
    fi

    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm

    python3 end_to_end_test.py \
        --concurrency 8 \
        -i http \
        --max-input-len 300 \
        --dataset ../dataset/mini_cnn_eval.json

    if [[ "$run_all_tests" == "true" ]]; then
        python3 end_to_end_test.py \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --dataset ../dataset/mini_cnn_eval.json
    fi

    python3 identity_test.py \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --concurrency 8 \
        -i http \
        --max-input-len 300 \
        dataset \
        --dataset ../dataset/mini_cnn_eval.json \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}

    if [[ "$run_all_tests" == "true" ]]; then
        python3 identity_test.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --num-requests 80 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE}

        python3 identity_test.py \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --request-rate -1 \
            --num-requests 100 \
            token-norm-dist \
            --input-mean 128 --input-stdev 0 \
            --output-mean 20 --output-stdev 0

    fi

    popd # tools/inflight_batcher_llm

    kill -9 ${SERVER_PID}
}

BATCHING_STRATEGIES=( "inflight_fused_batching")

MAX_NUM_SEQUENCES=( "" "4" "32" )
MAX_TOKENS_IN_KV_CACHES=( "" "2048" )
BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" "max_utilization")
KV_CACHE_FREE_GPU_MEM_FRACTIONS=( "0.2" "" )
ENABLE_TRT_OVERLAPS=( "false" "true" )

TRITON_MAX_BATCH_SIZE="128"
MAX_QUEUE_DELAY_MICROSECONDS="0"
MAX_BEAM_WIDTH="1"

if [ "$MODEL" = "gpt-ib" ]; then

    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    # -------------------------------
    # KV cache parameters
    # -------------------------------

    EXCLUDE_INPUT_IN_OUTPUT="false"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

    # We don't want to run all tests for all combination of parameters
    # Only run all tests for one combination of all parameters
    run_all_tests="true"

    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do

        run_cpp_backend_tests

        run_all_tests="false"
    done
    done
    done
    done
    done
    done #BATCHING STRATEGY

    MAX_NUM_SEQUENCE="${MAX_NUM_SEQUENCES[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_TRT_OVERLAP="${ENABLE_TRT_OVERLAPS[0]}"

    # -------------------------------
    # Exclude input in output test
    # -------------------------------
    EXCLUDE_INPUT_IN_OUTPUT="true"
    run_all_tests="false"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        run_cpp_backend_tests

    done
    EXCLUDE_INPUT_IN_OUTPUT="false"

    # -------------------------------
    #  Max queue delay microseconds
    # -------------------------------
    run_all_tests="false"
    MAX_QUEUE_DELAY_MICROSECONDS="1000000"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        run_cpp_backend_tests

    done
    MAX_QUEUE_DELAY_MICROSECONDS="0"

fi

run_cpp_streaming_backend_tests() {

    print_test_params

    # Because the runners are shared, the default value of 0.85 doesn't work, so skip
    # if max_tokens_in_kv_cache is also empty
    if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
        echo "Skipping..."
        continue
    fi

    rm -rf ./triton_repo
    cp -R all_models/inflight_batcher_llm triton_repo

    DECOUPLED_MODE="True"
    fill_triton_repo

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=triton_repo &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200

    EXCL_INPUT_IN_OUTPUT_FLAG=""
    [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"

    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --streaming \
        --check-output \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}

    if [[ "$run_all_tests" == "true" ]]; then
        # Stop request
        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} 2>&1 | tee output_w_stop

        grep "Got cancellation response" output_w_stop

        # Request cancellation
        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --stop-via-request-cancel \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} 2>&1 | tee output_w_stop

        grep "Request is cancelled" output_w_stop
    fi

    # End to end test
    python3 end_to_end_grpc_client.py \
        --output-len 10 --prompt "This is a test "
    popd

    kill -9 ${SERVER_PID}
}

if [ "$MODEL" = "gpt-ib-streaming" ]; then
    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    BATCHING_STRATEGY="inflight_fused_batching"
    EXCLUDE_INPUT_IN_OUTPUT="false"
    run_all_tests="true"

    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do

        run_cpp_streaming_backend_tests

        run_all_tests="false"
    done
    done
    done
    done
    done
fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    #Generate reference output
    pushd tensorrt_llm/examples/gpt

    # Input with virtual tokens:
    python3 run.py --max_output_len=8 --vocab_file=c-model/email_composition/fp16/1-gpu/tokenizer.model --prompt_table=email_composition.npy --input_tokens=input.csv --engine_dir ${ENGINE_PATH} --output_csv output_w_prompt.csv

    #Input w/o virtual tokens:
    echo "25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261" > input_wo_prompt.csv
    python3 run.py --max_output_len=8 --vocab_file=c-model/email_composition/fp16/1-gpu/tokenizer.model --input_tokens=input_wo_prompt.csv --engine_dir ${ENGINE_PATH} --output_csv output_wo_prompt.csv

    popd

    # Ptuning not enabled with V1 yet
    BATCHING_STRATEGIES=( "inflight_fused_batching" )

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        MAX_NUM_SEQUENCE=""
        MAX_TOKENS_IN_KV_CACHE=""
        BATCH_SCHEDULER_POLICIE="guaranteed_no_evict"
        KV_CACHE_FREE_GPU_MEM_FRACTION="0.2"
        ENABLE_TRT_OVERLAP="false"

        echo "----------------------------------"
        rm -rf ./triton_repo
        cp -R all_models/inflight_batcher_llm triton_repo

        # Modify config.pbtxt
        DECOUPLED_MODE="False"
        fill_triton_repo

        # Launch Triton Server
        /opt/tritonserver/bin/tritonserver \
            --model-repository=triton_repo &
        export SERVER_PID=$!
        wait_for_server_ready ${SERVER_PID} 1200

        # Test client
        pushd inflight_batcher_llm/client

        python3 inflight_batcher_llm_client.py --prompt-embedding-table ../../tensorrt_llm/examples/gpt/email_composition.npy --prompt-task-id 0 --input-tokens-csv ../../tensorrt_llm/examples/gpt/input.csv --output-tokens-csv ../../tensorrt_llm/examples/gpt/output_w_prompt.csv --check-output --request-output-len 8

        python3 inflight_batcher_llm_client.py --input-tokens-csv ../../tensorrt_llm/examples/gpt/input_wo_prompt.csv --output-tokens-csv ../../tensorrt_llm/examples/gpt/output_wo_prompt.csv --check-output --request-output-len 8

        popd # inflight_batcher_llm/client

        kill -9 ${SERVER_PID}
    done
fi
