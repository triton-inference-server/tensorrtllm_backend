#!/usr/bin/bash

MODEL=$1
TARGET_ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4
DRAFT_ENGINE_PATH=$5

set -ex
set -o pipefail
nvidia-smi
source tools/utils.sh

if [ "$MODEL" = "mistral" ] || [ "$MODEL" = "mistral-ib" ]; then
    MAX_ATTENTION_WINDOW_SIZE="2048"
    MAX_SEQUENCE_LEN="8704" # max_input_len + max_output_len
else
    MAX_ATTENTION_WINDOW_SIZE=""
    MAX_SEQUENCE_LEN="2048"
fi

if [ "$MODEL" = "gpt" ] || [ "$MODEL" = "opt" ] || [ "$MODEL" = "llama" ] || [ "$MODEL" = "gptj" ] || [ "$MODEL" = "mistral" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/gpt/tensorrt_llm/config.pbtxt engine_dir:${TARGET_ENGINE_PATH}
    python3 tools/fill_template.py -i all_models/gpt/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i all_models/gpt/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

    # Launch Triton Server
    mpirun --allow-run-as-root \
        -n 1 /opt/tritonserver/bin/tritonserver \
        --model-repository=all_models/gpt \
        --disable-auto-complete-config \
        --backend-config=python,shm-region-prefix-name=prefix0_ : &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200 ${TRITON_HTTP_PORT}

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

    # Benchmark Core Model
    python3 benchmark_core_model.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=http --mode=sync

    python3 benchmark_core_model.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=grpc --mode=sync

    python3 benchmark_core_model.py \
        --batch_size=8 --start_len=128 --output_len=20 \
        --protocol=http --mode=async

    python3 benchmark_core_model.py \
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
    echo "MAX_ATTENTION_WINDOW_SIZE: ${MAX_ATTENTION_WINDOW_SIZE}"
    echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
    echo "KV_CACHE_FREE_GPU_MEM_FRACTION: ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
    echo "ENABLE_TRT_OVERLAP: ${ENABLE_TRT_OVERLAP}"
    echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
    echo "TRITON_MAX_BATCH_SIZE: ${TRITON_MAX_BATCH_SIZE}"
    echo "MAX_QUEUE_DELAY_MICROSECONDS: ${MAX_QUEUE_DELAY_MICROSECONDS}"
    echo "MAX_BEAM_WIDTH: ${MAX_BEAM_WIDTH}"
    echo "ENABLE_KV_CACHE_REUSE: ${ENABLE_KV_CACHE_REUSE}"
    echo "E2E_MODEL_NAME: ${E2E_MODEL_NAME}"
    echo "ACCUMULATE_TOKEN: ${ACCUMULATE_TOKEN}"
    echo "BLS_INSTANCE_COUNT: ${BLS_INSTANCE_COUNT}"
    echo "PREPROCESSING_INSTANCE_COUNT: ${PREPROCESSING_INSTANCE_COUNT}"
    echo "POSTPROCESSING_INSTANCE_COUNT: ${POSTPROCESSING_INSTANCE_COUNT}"
    echo "run_all_tests: ${run_all_tests}"
    echo "----------------------------------"
}

fill_triton_repo () {

    echo "Filling triton repository at ${TRITON_REPO} with engine ${ENGINE_PATH}"

    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},max_num_sequences:${MAX_NUM_SEQUENCE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:${ENABLE_TRT_OVERLAP},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE}
    python3 tools/fill_template.py -i ${TRITON_REPO}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${PREPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${POSTPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},accumulate_tokens:${ACCUMULATE_TOKEN},bls_instance_count:${BLS_INSTANCE_COUNT}
}

kill_triton_server () {
    pkill -9 -f tritonserver
}

launch_triton_server () {

    print_test_params

    rm -rf ${TRITON_REPO}
    cp -R all_models/inflight_batcher_llm ${TRITON_REPO}

    # Modify config.pbtxt
    fill_triton_repo

    # Launch Triton Server
    /opt/tritonserver/bin/tritonserver \
        --model-repository=${TRITON_REPO} --http-port ${TRITON_HTTP_PORT} --grpc-port ${TRITON_GRPC_PORT} --metrics-port ${TRITON_METRICS_PORT} &
    export SERVER_PID=$!
    wait_for_server_ready ${SERVER_PID} 1200 ${TRITON_HTTP_PORT}
}

run_cpp_trtllm_backend_tests () {

    # test to run for all combinations of flags
    EXCL_INPUT_IN_OUTPUT_FLAG=""
    [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"

    # Test client
    pushd inflight_batcher_llm/client

    if [ $MAX_ATTENTION_WINDOW_SIZE ]; then
        # test using a longer input
        # TODO: Once we switch to using real weights, add `--check-output` arg
        python3 inflight_batcher_llm_client.py \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --input-tokens-csv='../../tools/dataset/long_input.csv' \
            --output-tokens-csv='../../tools/dataset/long_output.csv' \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            2>&1 | tee output_long_input

        # If no prompt in output, check that output sequence isn't an empty list of tokens
        if $EXCL_INPUT_IN_OUTPUT_FLAG; then
            grep -o "Output sequence starts with:  \[1, 3189, 28809, 28707, 7234, 574, 3441, 1236, 28723, 28705" output_long_input
        else
            grep -o "Output sequence\( starts with\)\?:\s*\[\([0-9]*\,\?\s\?\)*\]" output_long_input
        fi
    fi

    # testing output accuracy for real weights only
    CHECK_OUTPUT_FLAG=""
#    if [ $MODEL = "gpt-ib" ]; then
#        CHECK_OUTPUT_FLAG="--check-output"
#    fi

    python3 inflight_batcher_llm_client.py \
        ${CHECK_OUTPUT_FLAG} \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}

    if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then

        # testing output accuracy for real weights only
        if [[ $MODEL = "gpt-ib" ]]; then

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
        fi

        # test with request cancellation
        python3 inflight_batcher_llm_client.py \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --request-id 1 \
            2>&1 | tee output_w_stop
        grep "Got cancellation response" output_w_stop

        #test with return log probs
        python3 inflight_batcher_llm_client.py \
            --request-output-len=10 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --return-log-probs --top-k 2 \
            2>&1 | tee output_log_probs
    fi

    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm

    python3 benchmark_core_model.py \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --concurrency 8 \
        -i http \
        --max-input-len 300 \
        dataset \
        --dataset ../dataset/mini_cnn_eval.json \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}

    if [[ "$run_all_tests" == "true" ]]; then
        python3 benchmark_core_model.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --num-requests 80 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE}

        python3 benchmark_core_model.py \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --request-rate -1 \
            --num-requests 100 \
            token-norm-dist \
            --input-mean 128 --input-stdev 0 \
            --output-mean 20 --output-stdev 0

        python3 benchmark_core_model.py \
            -i grpc --max-input-len 1000 \
            --request-rate -1 \
            token-from-histogram --histogram-key example

    fi

    popd # tools/inflight_batcher_llm
}

run_cpp_e2e_backend_tests () {

    pushd inflight_batcher_llm/client

    # testing output accuracy for real weights only
    if [[ $MODEL = "gpt-ib" ]]; then

        python3 end_to_end_grpc_client.py \
            --output-len 10 --prompt "The only thing we have to fear is" | tee output_e2e
        grep "that the government will" output_e2e

        if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then
            # test with embedding bias
            python3 end_to_end_grpc_client.py \
                -o 10 \
                -p "The only thing we have to fear is"  \
                --embedding-bias-words " government" \
                --embedding-bias-weights -20 \
                2>&1 | tee output_w_bias
            grep -v "that the government will" output_w_bias
        fi
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

    popd # tools/inflight_batcher_llm
}

run_cpp_trtllm_streaming_backend_tests() {

    EXCL_INPUT_IN_OUTPUT_FLAG=""
    [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"

    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --streaming \
        --tokenizer-dir ${TOKENIZER_PATH} \
        --tokenizer-type ${TOKENIZER_TYPE}
#        --check-output \

    if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then
        # Stop request
        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --request-id 1 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} 2>&1 | tee output_w_stop

        grep "Got cancellation response" output_w_stop

        # Request cancellation
        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --request-id 1 \
            --stop-via-request-cancel \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} 2>&1 | tee output_w_stop

        grep "Request is cancelled" output_w_stop

        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --end-id 268 \
            --request-id 1 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --input-tokens-csv='../../tools/dataset/short_input_end_id.csv' \
            --output-tokens-csv='../../tools/dataset/short_output_end_id.csv' \
            --check-output
    fi

    popd
}

run_cpp_e2e_streaming_backend_tests() {

    OVERWRITE_OUTPUT_TEXT_FLAG=""
    [ "${ACCUMULATE_TOKEN}" = "true" ] && OVERWRITE_OUTPUT_TEXT_FLAG="--overwrite-output-text"

    pushd inflight_batcher_llm/client
    # End to end test
    python3 end_to_end_grpc_client.py \
        --streaming --output-len 10 --prompt "The only thing we have to fear is" | tee output_e2e
    grep "that the government will" output_e2e

    popd

    kill -9 ${SERVER_PID}
}

BATCHING_STRATEGIES=( "inflight_fused_batching" "v1" )
MAX_NUM_SEQUENCES=( "" "4" "32" )
MAX_TOKENS_IN_KV_CACHES=( "" $MAX_SEQUENCE_LEN )
BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" "max_utilization" )
KV_CACHE_FREE_GPU_MEM_FRACTIONS=( "0.2" "" )
ENABLE_TRT_OVERLAPS=( "false" "true" )

TRITON_MAX_BATCH_SIZE="128"
MAX_QUEUE_DELAY_MICROSECONDS="0"
MAX_BEAM_WIDTH="1"
ENABLE_KV_CACHE_REUSE="false"
E2E_MODEL_NAME="ensemble"
ACCUMULATE_TOKEN="false"
EXCLUDE_INPUT_IN_OUTPUT="false"
BLS_INSTANCE_COUNT="1"
PREPROCESSING_INSTANCE_COUNT="1"
POSTPROCESSING_INSTANCE_COUNT="1"
TRITON_REPO="triton_repo"
ENGINE_PATH=${TARGET_ENGINE_PATH}
TRITON_HTTP_PORT="8000"
TRITON_GRPC_PORT="8001"
TRITON_METRICS_PORT="8002"

if [ "$MODEL" = "gpt-ib" ] || [ "$MODEL" = "mistral-ib" ]; then

    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    # Non-streaming tests, decoupled is false
    DECOUPLED_MODE="False"

    # -------------------------------
    # Param sweep test
    # -------------------------------
    run_all_tests="true"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do

        # Because the runners are shared, the default value of 0.85 doesn't work, so skip
        # if max_tokens_in_kv_cache is also empty
        if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
            continue
        fi
        if [[ "${BATCHING_STRATEGY}" == "v1" && "${BATCH_SCHEDULER_POLICY}" == "max_utilization" ]]; then
            continue
        fi
	# For V1, batchScheduler currently cannot properly estimate kvCache usage
        if [[ "${BATCHING_STRATEGY}" == "v1" && "${MAX_TOKENS_IN_KV_CACHE}" != "" ]]; then
            continue
        fi

        launch_triton_server
        run_cpp_trtllm_backend_tests
        run_cpp_e2e_backend_tests
        kill_triton_server
        run_all_tests="false"
    done
    done
    done
    done
    done
    done
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
        launch_triton_server
        run_cpp_trtllm_backend_tests
        run_cpp_e2e_backend_tests
        kill_triton_server
    done
    EXCLUDE_INPUT_IN_OUTPUT="false"

    # -------------------------------
    #  Max queue delay microseconds
    # -------------------------------
    run_all_tests="false"
    MAX_QUEUE_DELAY_MICROSECONDS="1000000"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
        launch_triton_server
        run_cpp_trtllm_backend_tests
        run_cpp_e2e_backend_tests
        kill_triton_server
    done
    MAX_QUEUE_DELAY_MICROSECONDS="0"

    # -------------------------------
    #  Python BLS
    # -------------------------------
    ACCUMULATE_TOKENS=( "false" "true" )
    E2E_MODEL_NAMES=( "ensemble" "tensorrt_llm_bls" )
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for E2E_MODEL_NAME in "${E2E_MODEL_NAMES[@]}"; do
    for ACCUMULATE_TOKEN in "${ACCUMULATE_TOKENS[@]}"; do

        if [[ "${E2E_MODEL_NAME}" == "ensemble" && "${ACCUMULATE_TOKEN}" == "true" ]]; then
            continue
        fi
        launch_triton_server
        run_cpp_e2e_backend_tests
        kill_triton_server
    done
    done
    done
    E2E_MODEL_NAME="ensemble"
    ACCUMULATE_TOKEN="false"
fi

if [ "$MODEL" = "gpt-ib-streaming" ]; then
    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    DECOUPLED_MODE="True"
    run_all_tests="true"

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do

        # Because the runners are shared, the default value of 0.85 doesn't work, so skip
        # if max_tokens_in_kv_cache is also empty
        if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
            continue
        fi
        if [[ "${BATCHING_STRATEGY}" == "v1" && "${BATCH_SCHEDULER_POLICY}" == "max_utilization" ]]; then
            continue
        fi
	# For V1, batchScheduler currently cannot properly estimate kvCache usage
        if [[ "${BATCHING_STRATEGY}" == "v1" && "${MAX_TOKENS_IN_KV_CACHE}" != "" ]]; then
            continue
        fi

        launch_triton_server
        run_cpp_trtllm_streaming_backend_tests
        run_cpp_e2e_streaming_backend_tests
        kill_triton_server

        run_all_tests="false"
    done
    done
    done
    done
    done
    done
    MAX_NUM_SEQUENCE="${MAX_NUM_SEQUENCES[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_TRT_OVERLAP="${ENABLE_TRT_OVERLAPS[0]}"

    # --------------------
    # Python BLS test
    # --------------------
    ACCUMULATE_TOKENS=( "false" "true" )
    E2E_MODEL_NAMES=( "ensemble" "tensorrt_llm_bls" )
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for E2E_MODEL_NAME in "${E2E_MODEL_NAMES[@]}"; do
    for ACCUMULATE_TOKEN in "${ACCUMULATE_TOKENS[@]}"; do

        if [[ "${E2E_MODEL_NAME}" == "ensemble" && "${ACCUMULATE_TOKEN}" == "true" ]]; then
            continue
        fi
        launch_triton_server
        run_cpp_e2e_streaming_backend_tests
        kill_triton_server
    done
    done
    done
    E2E_MODEL_NAME="ensemble"
    ACCUMULATE_TOKEN="false"
fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    #Generate reference output
    pushd tensorrt_llm/examples/gpt

    # Input with virtual tokens:
    python3 ../run.py --max_output_len=8 --vocab_file=c-model/email_composition/fp16/1-gpu/tokenizer.model --prompt_table_path=email_composition.npy --input_file=input.csv --engine_dir ${TARGET_ENGINE_PATH} --output_csv output_w_prompt.csv

    #Input w/o virtual tokens:
    echo "25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261" > input_wo_prompt.csv
    python3 ../run.py --max_output_len=8 --vocab_file=c-model/email_composition/fp16/1-gpu/tokenizer.model --input_file=input_wo_prompt.csv --engine_dir ${TARGET_ENGINE_PATH} --output_csv output_wo_prompt.csv

    popd

    DECOUPLED_MODE="False"
    MAX_NUM_SEQUENCE="${MAX_NUM_SEQUENCES[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_TRT_OVERLAP="${ENABLE_TRT_OVERLAPS[0]}"

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        launch_triton_server

        # Test client
        pushd inflight_batcher_llm/client

        python3 inflight_batcher_llm_client.py --prompt-embedding-table ../../tensorrt_llm/examples/gpt/email_composition.npy --prompt-task-id 0 --input-tokens-csv ../../tensorrt_llm/examples/gpt/input.csv --output-tokens-csv ../../tensorrt_llm/examples/gpt/output_w_prompt.csv --check-output --request-output-len 8

        python3 inflight_batcher_llm_client.py --input-tokens-csv ../../tensorrt_llm/examples/gpt/input_wo_prompt.csv --output-tokens-csv ../../tensorrt_llm/examples/gpt/output_wo_prompt.csv --check-output --request-output-len 8

        popd # inflight_batcher_llm/client

        kill_triton_server
    done
fi

if [ "$MODEL" = "gpt-speculative-decoding" ]; then

    DECOUPLED_MODE="False"
    MAX_NUM_SEQUENCE="${MAX_NUM_SEQUENCES[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_TRT_OVERLAP="${ENABLE_TRT_OVERLAPS[0]}"

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

    	# Speculative decoding is not supported in V1
        if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
            continue
        fi

        TRITON_REPO="triton_repo"
        ENGINE_PATH=${TARGET_ENGINE_PATH}
        TRITON_HTTP_PORT="8000"
        TRITON_GRPC_PORT="8001"
        TRITON_METRICS_PORT="8002"
        ENABLE_KV_CACHE_REUSE="true"
        launch_triton_server

        TRITON_REPO="triton_repo_draft"
        ENGINE_PATH=${DRAFT_ENGINE_PATH}
        TRITON_HTTP_PORT="8003"
        TRITON_GRPC_PORT="8004"
        TRITON_METRICS_PORT="8005"
        # TODO(nkorobov): Draft model can benefit from enable KV cache.
        # Add --enable_context_fmha --use_paged_context_fmha to its build command
        ENABLE_KV_CACHE_REUSE="false"
        launch_triton_server

        # Test client
        pushd tools/inflight_batcher_llm

        python3 speculative_decoding_test.py --max-input-len 200 --dataset ../dataset/mini_cnn_eval.json --url-draft localhost:8004 --url-target localhost:8001

        popd # inflight_batcher_llm/client

        kill_triton_server
    done
fi
