#!/usr/bin/bash

MODEL=$1
DECODER_ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4
DRAFT_ENGINE_PATH=$5
TARGET_ENGINE_PATH=$6
ENCODER_ENGINE_PATH=$7
VISUAL_ENGINE_PATH=$8

set -ex
set -o pipefail
nvidia-smi
source tools/utils.sh

kill_triton_server () {
    pkill -9 -f trtllmExecutorWorker || true
    pkill -9 -f tritonserver
}

# Kill titonserver if it is still pending from previous test
kill_triton_server || true

if [ "$MODEL" = "mistral" ] || [ "$MODEL" = "mistral-ib" ] || [ "$MODEL" = "mistral-ib-mm" ]; then
    MAX_ATTENTION_WINDOW_SIZE="2048"
    MAX_SEQUENCE_LEN="8704" # max_input_len + max_output_len
elif [ "$MODEL" = "t5-ib" ] || [ "$MODEL" = "bart-ib" ]; then
    MAX_ATTENTION_WINDOW_SIZE=""
    MAX_SEQUENCE_LEN="4096" # for enc-dec, choose a sufficient size of max token in kv cache to avoid no free block error
else
    MAX_ATTENTION_WINDOW_SIZE=""
    MAX_SEQUENCE_LEN="2048"
fi

if [ "$MODEL" = "gpt" ] || [ "$MODEL" = "opt" ] || [ "$MODEL" = "llama" ] || [ "$MODEL" = "gptj" ] || [ "$MODEL" = "mistral" ]; then
    # Modify config.pbtxt
    python3 tools/fill_template.py -i all_models/gpt/tensorrt_llm/config.pbtxt engine_dir:${DECODER_ENGINE_PATH}
    python3 tools/fill_template.py -i all_models/gpt/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH}
    python3 tools/fill_template.py -i all_models/gpt/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH}

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
        --tokenizer_dir ${TOKENIZER_PATH}

    python3 client.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc \
        --tokenizer_dir ${TOKENIZER_PATH}

    # Async Client
    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=http \
        --tokenizer_dir ${TOKENIZER_PATH}

    python3 client_async.py \
        --text="Born in north-east France, Soyer trained as a" \
        --output_len=10 \
        --protocol=grpc \
        --tokenizer_dir ${TOKENIZER_PATH}

    # End to end test
    python3 end_to_end_test.py \
        --tokenizer_dir ${TOKENIZER_PATH}

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
    echo "BACKEND: ${BACKEND}"
    echo "BATCHING_STRATEGY: ${BATCHING_STRATEGY}"
    echo "MAX_TOKENS_IN_KV_CACHE: ${MAX_TOKENS_IN_KV_CACHE}"
    echo "MAX_ATTENTION_WINDOW_SIZE: ${MAX_ATTENTION_WINDOW_SIZE}"
    echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
    echo "KV_CACHE_FREE_GPU_MEM_FRACTION: ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
    echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
    echo "TRITON_MAX_BATCH_SIZE: ${TRITON_MAX_BATCH_SIZE}"
    echo "MAX_QUEUE_DELAY_MICROSECONDS: ${MAX_QUEUE_DELAY_MICROSECONDS}"
    echo "MAX_BEAM_WIDTH: ${MAX_BEAM_WIDTH}"
    echo "ENABLE_KV_CACHE_REUSE: ${ENABLE_KV_CACHE_REUSE}"
    echo "E2E_MODEL_NAME: ${E2E_MODEL_NAME}"
    echo "TENSORRT_LLM_MODEL_NAME: ${TENSORRT_LLM_MODEL_NAME}"
    echo "TENSORRT_LLM_TARGET_MODEL_NAME: ${TENSORRT_LLM_TARGET_MODEL_NAME}"
    echo "TENSORRT_LLM_DRAFT_MODEL_NAME: ${TENSORRT_LLM_DRAFT_MODEL_NAME}"
    echo "ACCUMULATE_TOKEN: ${ACCUMULATE_TOKEN}"
    echo "BLS_INSTANCE_COUNT: ${BLS_INSTANCE_COUNT}"
    echo "PREPROCESSING_INSTANCE_COUNT: ${PREPROCESSING_INSTANCE_COUNT}"
    echo "POSTPROCESSING_INSTANCE_COUNT: ${POSTPROCESSING_INSTANCE_COUNT}"
    echo "NORMALIZE_LOG_PROBS: ${NORMALIZE_LOG_PROBS}"
    echo "ENABLE_CHUNKED_CONTEXT: ${ENABLE_CHUNKED_CONTEXT}"
    echo "GPU_DEVICE_IDS: ${GPU_DEVICE_IDS}"
    echo "DECODING_MODE: ${DECODING_MODE}"
    echo "MAX_QUEUE_SIZE: ${MAX_QUEUE_SIZE}"
    echo "run_all_tests: ${run_all_tests}"
    echo "----------------------------------"
}

fill_triton_repo () {

    if [ "${DRAFT_ENGINE_PATH}" != "" ] && [ "${DRAFT_ENGINE_PATH}" != "skip" ]; then
        cp -R ${TRITON_REPO}/tensorrt_llm ${TRITON_REPO}/tensorrt_llm_draft
        sed -i 's/name: "tensorrt_llm"/name: "tensorrt_llm_draft"/g' ${TRITON_REPO}/tensorrt_llm_draft/config.pbtxt
    fi

    if [ "${TARGET_ENGINE_PATH}" != "" ] && [ "${TARGET_ENGINE_PATH}" != "skip" ]; then
        cp -R ${TRITON_REPO}/tensorrt_llm ${TRITON_REPO}/tensorrt_llm_target
        sed -i 's/name: "tensorrt_llm"/name: "tensorrt_llm_target"/g' ${TRITON_REPO}/tensorrt_llm_target/config.pbtxt
    fi

    echo "Filling triton repository at ${TRITON_REPO}/tensorrt_llm with engine ${DECODER_ENGINE_PATH}"
    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm/config.pbtxt triton_backend:${BACKEND},engine_dir:${DECODER_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE}

    python3 tools/fill_template.py -i ${TRITON_REPO}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${PREPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${POSTPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}

    if [ "${DRAFT_ENGINE_PATH}" != "" ] && [ "${DRAFT_ENGINE_PATH}" != "skip" ] && [ "${TARGET_ENGINE_PATH}" != "" ] && [ "${TARGET_ENGINE_PATH}" != "skip" ]; then
        python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},accumulate_tokens:${ACCUMULATE_TOKEN},bls_instance_count:${BLS_INSTANCE_COUNT},tensorrt_llm_model_name:${TENSORRT_LLM_TARGET_MODEL_NAME},tensorrt_llm_draft_model_name:${TENSORRT_LLM_DRAFT_MODEL_NAME}
    else
        python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},accumulate_tokens:${ACCUMULATE_TOKEN},bls_instance_count:${BLS_INSTANCE_COUNT},tensorrt_llm_model_name:${TENSORRT_LLM_MODEL_NAME},tensorrt_llm_draft_model_name:${TENSORRT_LLM_MODEL_NAME}
    fi

    if [ "${DRAFT_ENGINE_PATH}" != "" ] && [ "${DRAFT_ENGINE_PATH}" != "skip" ]; then
        echo "Filling triton repository at ${TRITON_REPO}/tensorrt_llm_draft with engine ${DRAFT_ENGINE_PATH}"
        python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_draft/config.pbtxt triton_backend:${BACKEND},engine_dir:${DRAFT_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE}
    fi

    if [ "${TARGET_ENGINE_PATH}" != "" ] && [ "${TARGET_ENGINE_PATH}" != "skip" ]; then
        echo "Filling triton repository at ${TRITON_REPO}/tensorrt_llm_target with engine ${TARGET_ENGINE_PATH}"
        python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_target/config.pbtxt triton_backend:${BACKEND},engine_dir:${TARGET_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE}
    fi

    if [ "${ENCODER_ENGINE_PATH}" != "" ] && [ "${ENCODER_ENGINE_PATH}" != "skip" ]; then
        python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm/config.pbtxt encoder_engine_dir:${ENCODER_ENGINE_PATH}
    fi

    if [ "${VISUAL_ENGINE_PATH}" != "" ] && [ "${VISUAL_ENGINE_PATH}" != "skip" ]; then
        python3 tools/fill_template.py -i ${TRITON_REPO}/preprocessing/config.pbtxt visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${DECODER_ENGINE_PATH}
    fi
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

    STREAMING_FLAG=""
    [ "${STREAMING}" = "true" ] && STREAMING_FLAG="--streaming"

    # Test client
    pushd inflight_batcher_llm/client

    if [ $MAX_ATTENTION_WINDOW_SIZE ]; then
        # test using a longer input
        # TODO: Once we switch to using real weights, add `--check-output` arg
        python3 inflight_batcher_llm_client.py \
            ${STREAMING_FLAG} \
            --tokenizer-dir ${TOKENIZER_PATH} \
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
    if [ $MODEL = "gpt-ib" ]; then
        CHECK_OUTPUT_FLAG="--check-output"
    fi

    python3 inflight_batcher_llm_client.py \
        ${STREAMING_FLAG} \
        ${CHECK_OUTPUT_FLAG} \
        ${EXCL_INPUT_IN_OUTPUT_FLAG} \
        --tokenizer-dir ${TOKENIZER_PATH}

    #Check that metrics work as expected by looking at number of successful requests for tensorrt_llm
    num_success=$(curl localhost:${TRITON_METRICS_PORT}/metrics 2>&1 |  grep nv_inference_request_success\{model=\"tensorrt_llm\" | cut -d " " -f 2)
    if (( num_success <= 0 )); then
      exit 1
    else
      echo "Number of successful requests: $num_success"
    fi

    if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then

        # testing output accuracy for real weights only
        if [[ $MODEL = "gpt-ib" ]]; then

            test_stop_words ()
            {
                PROMPT="The only thing we have to fear is"
                OUTLEN=10

                ORIGINAL_OUTPUT=$(python3 end_to_end_grpc_client.py ${STREAMING_FLAG} -o ${OUTLEN} -p "${PROMPT}" 2>&1 | tail -n 1)
                echo "original output"
                echo $ORIGINAL_OUTPUT
                # should be something like "[...] that the government will [...]"

                # examples of stop words that won't affect generation
                # "government" isn't tokenized like " government"
                # " that the public" doesn't match entirely the generated string
                TEST_OUTPUT=$(python3 end_to_end_grpc_client.py ${STREAMING_FLAG} -o ${OUTLEN} -p "${PROMPT}" --stop-words "government" " that the public" 2>&1 | tail -n 1)
                [[ "${ORIGINAL_OUTPUT}" == "${TEST_OUTPUT}" ]]

                # check that output finishes at "government"
                TEST_OUTPUT=$(python3 end_to_end_grpc_client.py ${STREAMING_FLAG} -o ${OUTLEN} -p "${PROMPT}" --stop-words " lorem" " government" 2>&1 | tail -n 1)
                [[ "${TEST_OUTPUT}" == *"government" ]]
                TEST_OUTPUT=$(python3 end_to_end_grpc_client.py ${STREAMING_FLAG} -o ${OUTLEN} -p "${PROMPT}" --stop-words " that the government" 2>&1 | tail -n 1)
                [[ "${TEST_OUTPUT}" == *"government" ]]
            }
            test_stop_words
        fi

        # Stop request
        python3 inflight_batcher_llm_client.py \
            ${STREAMING_FLAG} \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --request-id 1 \
            2>&1 | tee output_w_stop
        grep "Got cancellation response" output_w_stop

        if [[ "${STREAMING}" == "true" ]]; then
            # Request cancellation
            python3 inflight_batcher_llm_client.py \
                ${EXCL_INPUT_IN_OUTPUT_FLAG} \
                --streaming \
                --request-output-len=128 \
                --stop-after-ms 100 \
                --request-id 1 \
                --stop-via-request-cancel \
                --tokenizer-dir ${TOKENIZER_PATH} 2>&1 | tee output_w_stop

            grep "Request is cancelled" output_w_stop
        fi

        if [[ -n "${1}" && -n "${2}" && -n "${3}" ]]; then
            python3 inflight_batcher_llm_client.py \
                ${EXCL_INPUT_IN_OUTPUT_FLAG} \
                ${STREAMING_FLAG} \
                --request-output-len=128 \
                --end-id $3 \
                --request-id 1 \
                --tokenizer-dir ${TOKENIZER_PATH} \
                --input-tokens-csv=$1 \
                --output-tokens-csv=$2 \
                --check-output
        fi

        #test with return log probs
        python3 inflight_batcher_llm_client.py \
            ${STREAMING_FLAG} \
            --request-output-len=10 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --return-log-probs --top-k 2 \
            2>&1 | tee output_log_probs

        #test with string request id
        python3 inflight_batcher_llm_client.py \
            ${STREAMING_FLAG} \
            ${CHECK_OUTPUT_FLAG} \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --request-id my_request 2>&1 | tee output_str_request

        # Test triton metrics are present and have non-zero values (when applicable).
        TRITON_METRICS_LOG="triton_metrics.out"
        curl localhost:${TRITON_METRICS_PORT}/metrics -o ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_request_metrics\{model="tensorrt_llm",request_type="context",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_request_metrics\{model="tensorrt_llm",request_type="scheduled",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_request_metrics\{model="tensorrt_llm",request_type="max",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_request_metrics\{model="tensorrt_llm",request_type="active",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_runtime_memory_metrics\{memory_type="pinned",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_runtime_memory_metrics\{memory_type="gpu",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_runtime_memory_metrics\{memory_type="cpu",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_kv_cache_block_metrics\{kv_cache_block_type="tokens_per",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_kv_cache_block_metrics\{kv_cache_block_type="used",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_kv_cache_block_metrics\{kv_cache_block_type="free",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_kv_cache_block_metrics\{kv_cache_block_type="max",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{model="tensorrt_llm",v1_specific_metric="num_ctx_tokens",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{model="tensorrt_llm",v1_specific_metric="num_gen_tokens",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{model="tensorrt_llm",v1_specific_metric="empty_gen_slots",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
        else
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{inflight_batcher_specific_metric="paused_requests",model="tensorrt_llm",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{inflight_batcher_specific_metric="micro_batch_id",model="tensorrt_llm",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
          grep -E 'nv_trt_llm_inflight_batcher_metrics\{inflight_batcher_specific_metric="total_context_tokens",model="tensorrt_llm",version="1"\} [0-9]+$' ${TRITON_METRICS_LOG}
        fi
        grep -E 'nv_trt_llm_general_metrics\{general_type="iteration_counter",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        grep -E 'nv_trt_llm_general_metrics\{general_type="timestamp",model="tensorrt_llm",version="1"\} [1-9][0-9]*$' ${TRITON_METRICS_LOG}
        rm ${TRITON_METRICS_LOG}
    fi

    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm

    # HTTP client cannot be used with decoupled mode.
    if [[ "${DECOUPLED_MODE}" == "False" ]]; then
        python3 benchmark_core_model.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --concurrency 8 \
            -i http \
            --max-input-len 300 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer-dir ${TOKENIZER_PATH}
    fi

    if [[ "$run_all_tests" == "true" ]]; then
        # Note: streaming flag is not set to 1 for these benchmarks regardless
        # of the value of $STREAMING.
        DECOUPLED_FLAG=""
        [ "${DECOUPLED_MODE}" = "True" ] && DECOUPLED_FLAG="--decoupled"

        python3 benchmark_core_model.py \
            ${DECOUPLED_FLAG} \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --num-requests 80 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer-dir ${TOKENIZER_PATH}

        # Performance check.
        python3 benchmark_core_model.py \
            ${DECOUPLED_FLAG} \
            ${CHECK_PERF_JSON_ARGS} \
            --check-perf-key ${MODEL}-${BACKEND} \
            --check-perf-rtol 0.05 \
            --check-perf-atol 50 \
            --concurrency 8 \
            -i grpc \
            --max-input-len 300 \
            --request-rate -1 \
            --num-requests 1000 \
            token-norm-dist \
            --input-mean 128 --input-stdev 0 \
            --output-mean 20 --output-stdev 0

        python3 benchmark_core_model.py \
            ${DECOUPLED_FLAG} \
            -i grpc --max-input-len 1000 \
            --request-rate -1 \
            token-from-histogram --histogram-key example

    fi

    popd # tools/inflight_batcher_llm
}

run_cpp_e2e_backend_tests () {

    STREAMING_FLAG=""
    [ "${STREAMING}" = "true" ] && STREAMING_FLAG="--streaming"

    pushd inflight_batcher_llm/client

    # testing output accuracy for real weights only
    if [[ $MODEL = "gpt-ib" || $MODEL = "gpt-ib-streaming" ]]; then

        python3 end_to_end_grpc_client.py \
            ${STREAMING_FLAG} \
            --output-len 10 --prompt "The only thing we have to fear is" \
            --model-name "$E2E_MODEL_NAME" | tee output_e2e
        grep "that the government will" output_e2e

        if [[ "$run_all_tests" == "true" && "$BATCHING_STRATEGY" == "inflight_fused_batching" ]]; then
            # test with embedding bias
            python3 end_to_end_grpc_client.py \
                ${STREAMING_FLAG} \
                -o 10 \
                -p "The only thing we have to fear is"  \
                --embedding-bias-words " government" \
                --embedding-bias-weights -20 \
                2>&1 | tee output_w_bias
            grep -v "that the government will" output_w_bias

            #Only run batched test in streaming for now since it requires decoupled mode
            if [[ "$STREAMING" == "true" ]]; then
                # test with batched requests
                python3 end_to_end_grpc_client.py \
                    ${STREAMING_FLAG} \
                    -o 5 \
                    -p '["This is a test","I want you to","The cat is"]'  \
                    --batch-inputs --check-outputs --expected-outputs '[" of the power of the"," know that I am not"," a very good cat."]'
            fi
        fi
    fi

    popd # inflight_batcher_llm/client

    # End to end test
    pushd tools/inflight_batcher_llm
    # end_to_end_test.py doesn't support streaming
    if [[ "${STREAMING}" == "false" ]]; then
        python3 end_to_end_test.py \
            --concurrency 8 \
            -i http \
            --max-input-len 200 \
            --test-bls \
            --dataset ../dataset/mini_cnn_eval.json

        if [[ "$run_all_tests" == "true" ]]; then
            python3 end_to_end_test.py \
                --concurrency 8 \
                -i grpc \
                --max-input-len 200 \
                --test-bls \
                --dataset ../dataset/mini_cnn_eval.json
        fi
    fi

    popd # tools/inflight_batcher_llm
}

run_cpp_trtllm_queue_size_tests () {
    # Test client
    echo "25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261" > input.csv
    pushd tools/inflight_batcher_llm
    EXTRA_FLAGS=""
    if [[ "${DECOUPLED_MODE}" == "True" ]]; then
        EXTRA_FLAGS="-p grpc -u localhost:8001"
    fi
    python3 test_max_queue_size.py --input-tokens-csv ../../input.csv --request-output-len 256 --num-requests 100 ${EXTRA_FLAGS}

    popd # tools/inflight_batcher_llm
}

BACKENDS=( "tensorrtllm" "python" )
BATCHING_STRATEGIES=( "inflight_fused_batching" "v1" )
MAX_TOKENS_IN_KV_CACHES=( "" $MAX_SEQUENCE_LEN )
BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" "max_utilization" )
KV_CACHE_FREE_GPU_MEM_FRACTIONS=( "0.2" "" )
ENABLE_CHUNKED_CONTEXTS=( "false" "true" )

BACKEND="tensorrtllm"
TRITON_MAX_BATCH_SIZE="128"
MAX_QUEUE_DELAY_MICROSECONDS="0"
MAX_BEAM_WIDTH="1"
ENABLE_KV_CACHE_REUSE="false"
E2E_MODEL_NAME="ensemble"
TENSORRT_LLM_MODEL_NAME="tensorrt_llm"
TENSORRT_LLM_DRAFT_MODEL_NAME="tensorrt_llm_draft"
TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm_target"
ACCUMULATE_TOKEN="false"
EXCLUDE_INPUT_IN_OUTPUT="false"
BLS_INSTANCE_COUNT="1"
PREPROCESSING_INSTANCE_COUNT="1"
POSTPROCESSING_INSTANCE_COUNT="1"
NORMALIZE_LOG_PROBS="true"
TRITON_REPO="triton_repo"
TRITON_HTTP_PORT="8000"
TRITON_GRPC_PORT="8001"
TRITON_METRICS_PORT="8002"
GPU_DEVICE_IDS=""
DECODING_MODE="top_k_top_p"
MAX_QUEUE_SIZE="0"

if [ "$MODEL" = "gpt-ib" ] || [ "$MODEL" = "mistral-ib" ] || [ "$MODEL" = "mistral-ib-mm" ]; then

    # Non-streaming tests, decoupled is false
    DECOUPLED_MODE="False"
    STREAMING="false"

    # -------------------------------
    # Param sweep test
    # -------------------------------
    run_all_tests="true"
    for BACKEND in "${BACKENDS[@]}"; do
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_CHUNKED_CONTEXT in "${ENABLE_CHUNKED_CONTEXTS[@]}"; do

        # Because the runners are shared, the default value of 0.9 doesn't work, so skip
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
        # mistral is built without chunked context support
        if [[ "$MODEL" = "mistral-ib" && "${ENABLE_CHUNKED_CONTEXT}" == "true" ]]; then
            continue
        fi
        if [[ "$MODEL" = "mistral-ib-mm" && "${ENABLE_CHUNKED_CONTEXT}" == "true" ]]; then
            continue
        fi
        # The python backend currently requires decoupled mode.
        if [[ "${BACKEND}" == "python" && "${DECOUPLED_MODE}" == "False" ]]; then
           continue
        fi

        if [[ "$MODEL" = "mistral-ib-mm" ]]; then
            export TRTLLM_ORCHESTRATOR=1
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
    BACKEND="${BACKENDS[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_CHUNKED_CONTEXT="${ENABLE_CHUNKED_CONTEXTS[0]}"

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
    #  Max queue size
    # -------------------------------
    run_all_tests="false"
    MAX_QUEUE_SIZE="6"
    TRITON_MAX_BATCH_SIZE="1"
    BATCHING_STRATEGY="inflight_fused_batching"

    for BACKEND in "${BACKENDS[@]}"; do
        if [[ "${BACKEND}" == "python" ]]; then
            DECOUPLED_MODE="True"
        fi
        launch_triton_server
        run_cpp_trtllm_queue_size_tests
        kill_triton_server
        DECOUPLED_MODE="False"
    done

    MAX_QUEUE_SIZE="0"
    TRITON_MAX_BATCH_SIZE="128"
    BACKEND="${BACKENDS[0]}"

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

    DECOUPLED_MODE="True"
    STREAMING="true"
    run_all_tests="true"

    for BACKEND in "${BACKENDS[@]}"; do
        run_all_tests="true"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_CHUNKED_CONTEXT in "${ENABLE_CHUNKED_CONTEXTS[@]}"; do

        # Because the runners are shared, the default value of 0.9 doesn't work, so skip
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
        # The python backend currently requires decoupled mode.
        if [[ "${BACKEND}" == "python" && "${DECOUPLED_MODE}" == "False" ]]; then
           continue
        fi

        launch_triton_server
        run_cpp_trtllm_backend_tests '../../tools/dataset/short_input_end_id.csv' '../../tools/dataset/short_output_end_id.csv' 268
        run_cpp_e2e_backend_tests
        kill_triton_server

        run_all_tests="false"
    done
    done
    done
    done
    done
    done
    BACKEND="${BACKENDS[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    ENABLE_CHUNKED_CONTEXT="${ENABLE_CHUNKED_CONTEXTS[0]}"

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
        run_cpp_e2e_backend_tests
        kill_triton_server
    done
    done
    done
    E2E_MODEL_NAME="ensemble"
    ACCUMULATE_TOKEN="false"
fi

if [ "$MODEL" = "gpt-ib-speculative-decoding-bls" ]; then
    # --------------------
    # Python BLS test
    # --------------------
    DECOUPLED_MODE="False"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    USE_DRAFT_LOGITS_VALUES=( "true" "false" )

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for USE_DRAFT_LOGITS in "${USE_DRAFT_LOGITS_VALUES[@]}"; do


        if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
            continue
        fi
        draft_args="--num-draft-tokens=5"
        if [[ "${USE_DRAFT_LOGITS}" == "true" ]]; then
            # with draft logit compare the outputs are not deterministic so we just
            draft_args="--num-draft-tokens=5 --return-generation-logits --use-draft-logits --disable-output-comparison"
        fi
        ENABLE_KV_CACHE_REUSE="true"
        launch_triton_server

        # Test client
        pushd tools/inflight_batcher_llm

        python3 speculative_decoding_test.py \
            --max-input-len 200 \
            --dataset ../dataset/mini_cnn_eval_spec_decoding.json \
            --url-target=localhost:8001 \
            --url-draft=localhost:8001 \
            --url-control=localhost:8001 \
            --draft-tensorrt-llm-model-name="${TENSORRT_LLM_DRAFT_MODEL_NAME}" \
            --target-tensorrt-llm-model-name="${TENSORRT_LLM_TARGET_MODEL_NAME}" \
            --bls-speculative-tensorrt-llm-model-name="tensorrt_llm_bls" \
            --execute-bls-speculative-decoding \
            ${draft_args} \
            --verbose

        popd # inflight_batcher_llm/client

        kill_triton_server
    done
    done
fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    #Generate reference output
    pushd tensorrt_llm/examples/gpt

    # Input with virtual tokens:
    python3 ../run.py \
        --max_output_len=8 \
        --vocab_file=c-model/email_composition/fp16/tokenizer.model \
        --prompt_table_path=email_composition.npy \
        --input_file=input.csv \
        --engine_dir ${DECODER_ENGINE_PATH} \
        --output_csv output_w_prompt.csv

    #Input w/o virtual tokens:
    echo "25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261" > input_wo_prompt.csv
    python3 ../run.py \
        --max_output_len=8 \
        --vocab_file=c-model/email_composition/fp16/tokenizer.model \
        --input_file=input_wo_prompt.csv \
        --engine_dir ${DECODER_ENGINE_PATH} \
        --output_csv output_wo_prompt.csv

    popd

    DECOUPLED_MODE="False"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"

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

if [ "$MODEL" = "gpt-2b-ib-lora" ]; then

    #Generate reference output
    pushd tensorrt_llm/examples/gpt

    # Input with virtual tokens:
    python3 ../run.py \
        --max_output_len=8 \
        --lora_dir=gpt2b_lora-900.nemo \
        --lora_ckpt_source nemo \
        --lora_task_uids 0 \
        --engine_dir ${DECODER_ENGINE_PATH} \
        --input_file=input.csv \
        --output_csv output.csv \
        --use_py_session

    popd

    DECOUPLED_MODE="False"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        # LoRA is not supported in V1
        if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
            continue
        fi

        launch_triton_server

        # Test client
        pushd inflight_batcher_llm/client

        python3 inflight_batcher_llm_client.py \
            --input-tokens-csv ../../tensorrt_llm/examples/gpt/input.csv \
            --output-tokens-csv ../../tensorrt_llm/examples/gpt/output.csv \
            --check-output --request-output-len 8 \
            --lora-path ../../tensorrt_llm/examples/gpt/gpt-2b-lora-train-900 \
            --lora-task-id 12345

        python3 inflight_batcher_llm_client.py \
            --input-tokens-csv ../../tensorrt_llm/examples/gpt/input.csv \
            --output-tokens-csv ../../tensorrt_llm/examples/gpt/output.csv \
            --check-output --request-output-len 8 \
            --lora-task-id 12345

        popd # inflight_batcher_llm/client

        kill_triton_server
    done
fi

if [ "$MODEL" = "gpt-speculative-decoding" ]; then

    DECOUPLED_MODE="False"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        # Speculative decoding is not supported in V1
        if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
            continue
        fi

        TRITON_HTTP_PORT="8000"
        TRITON_GRPC_PORT="8001"
        TRITON_METRICS_PORT="8002"
        ENABLE_KV_CACHE_REUSE="true"
        launch_triton_server

        TRITON_HTTP_PORT="8003"
        TRITON_GRPC_PORT="8004"
        TRITON_METRICS_PORT="8005"
        # TODO(nkorobov): Draft model can benefit from enable KV cache.
        # Add --enable_context_fmha --use_paged_context_fmha to its build command
        ENABLE_KV_CACHE_REUSE="false"
        launch_triton_server

        # Test client
        pushd tools/inflight_batcher_llm

        python3 speculative_decoding_test.py \
            --max-input-len 200 \
            --dataset ../dataset/mini_cnn_eval_spec_decoding.json \
            --url-draft localhost:8004 \
            --url-target localhost:8001 \
            --url-control localhost:8001 \
            --draft-tensorrt-llm-model-name="${TENSORRT_LLM_DRAFT_MODEL_NAME}" \
            --target-tensorrt-llm-model-name="${TENSORRT_LLM_TARGET_MODEL_NAME}" \
            --verbose

        popd # inflight_batcher_llm/client

        kill_triton_server
    done
fi

if [ "$MODEL" = "gpt-gather-logits" ]; then

    if [ "${DRAFT_ENGINE_PATH}" == "" ]; then
        # normal gather logits test
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

            python3 inflight_batcher_llm_client.py \
                --tokenizer-dir ${TOKENIZER_PATH} \
                --return-context-logits \
                --return-generation-logits

            python3 inflight_batcher_llm_client.py \
                --tokenizer-dir ${TOKENIZER_PATH}
            popd # inflight_batcher_llm/client

            pushd tools/inflight_batcher_llm
            python3 end_to_end_test.py \
                -i http \
                --max-input-len 192 \
                --return-context-logits \
                --return-generation-logits \
                --dataset ../dataset/mini_cnn_eval.json

            python3 end_to_end_test.py \
                -i http \
                --max-input-len 192 \
                --dataset ../dataset/mini_cnn_eval.json

            popd # tools/inflight_batcher_llm

            kill_triton_server
        done

    else
        # test with speculative decoding
        # speculative decoding return draft model draft token logits
        # and target model accepted token logits

        DECOUPLED_MODE="False"
        MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
        BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
        KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
        ENABLE_TRT_OVERLAP="${ENABLE_TRT_OVERLAPS[0]}"

        for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

            # Speculative decoding is not supported in V1
            if [[ "${BATCHING_STRATEGY}" == "v1" ]]; then
                continue
            fi

            TRITON_HTTP_PORT="8000"
            TRITON_GRPC_PORT="8001"
            TRITON_METRICS_PORT="8002"
            ENABLE_KV_CACHE_REUSE="true"
            launch_triton_server

            TRITON_HTTP_PORT="8003"
            TRITON_GRPC_PORT="8004"
            TRITON_METRICS_PORT="8005"
            # TODO(nkorobov): Draft model can benefit from enable KV cache.
            # Add --enable_context_fmha --use_paged_context_fmha to its build command
            ENABLE_KV_CACHE_REUSE="false"
            launch_triton_server

            # Test client
            pushd tools/inflight_batcher_llm

            python3 speculative_decoding_test.py \
                --max-input-len 128 \
                --dataset ../dataset/mini_cnn_eval_spec_decoding.json \
                --url-draft localhost:8004 \
                --url-target localhost:8001 \
                --url-control localhost:8001 \
                --draft-tensorrt-llm-model-name="${TENSORRT_LLM_DRAFT_MODEL_NAME}" \
                --target-tensorrt-llm-model-name="${TENSORRT_LLM_TARGET_MODEL_NAME}" \
                --num-draft-tokens=5 \
                --return-target-model-accepted-token-logits \
                --return-draft-model-draft-logits \
                --verbose

            popd # inflight_batcher_llm/client

            kill_triton_server
        done
    fi

fi

if [ "$MODEL" = "medusa" ]; then
    # To make sure that torch is not a dependency for C++ backend
    # pip3 uninstall -y torch

    # Test streaming
    DECOUPLED_MODE="True"
    STREAMING="true"
    run_all_tests="true"

    MAX_NUM_SEQUENCE="${MAX_NUM_SEQUENCES[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"
    BATCHING_STRATEGY="${BATCHING_STRATEGIES[0]}"
    DECODING_MODE="medusa"

    END_ID_MEDUSA=1284
    MEDUSA_INPUT_IDS_PATH='../../tools/dataset/short_input_end_id_medusa.csv'
    MEDUSA_OUTPUT_IDS_PATH='../../tools/dataset/short_output_end_id_medusa.csv'

    launch_triton_server
    run_cpp_trtllm_backend_tests ${MEDUSA_INPUT_IDS_PATH} ${MEDUSA_OUTPUT_IDS_PATH} ${END_ID_MEDUSA}
    kill_triton_server
    # FIXME: grpc e2e test returns different result (because it is Medusa and not GPT) and has some problems with spaces

    # Test non-streaming
    DECOUPLED_MODE="False"
    launch_triton_server
    # Test client
    pushd inflight_batcher_llm/client
    python3 inflight_batcher_llm_client.py \
            --request-output-len=128 \
            --end-id ${END_ID_MEDUSA} \
            --request-id 1 \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --input-tokens-csv ${MEDUSA_INPUT_IDS_PATH} \
            --output-tokens-csv ${MEDUSA_OUTPUT_IDS_PATH} \
            --check-output
    popd # inflight_batcher_llm/client
    kill_triton_server
fi

if [ "$MODEL" = "bart-ib" ] || [ "$MODEL" = "t5-ib" ]; then

    # Non-streaming tests, decoupled is false
    DECOUPLED_MODE="False"
    STREAMING="false"

    # enc-dec models only support inflight_fused_batching, with chunked context disabled
    CHECK_PERF_JSON_ARGS=""
    BATCHING_STRATEGY="inflight_fused_batching"
    ENABLE_CHUNKED_CONTEXT="false"

    # -------------------------------
    # Param sweep test
    # -------------------------------
    run_all_tests="true"
    for BACKEND in "${BACKENDS[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
        # Because the runners are shared, the default value of 0.9 doesn't work, so skip
        # if max_tokens_in_kv_cache is also empty
        if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
            continue
        fi
        # The python backend currently requires decoupled mode.
        if [[ "${BACKEND}" == "python" && "${DECOUPLED_MODE}" == "False" ]]; then
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
    BACKEND="${BACKENDS[0]}"
    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"

    # -------------------------------
    # Exclude input in output test
    # -------------------------------
    EXCLUDE_INPUT_IN_OUTPUT="true"
    run_all_tests="false"
    launch_triton_server
    run_cpp_trtllm_backend_tests
    run_cpp_e2e_backend_tests
    kill_triton_server
    EXCLUDE_INPUT_IN_OUTPUT="false"

    # -------------------------------
    #  Max queue delay microseconds
    # -------------------------------
    run_all_tests="false"
    MAX_QUEUE_DELAY_MICROSECONDS="1000000"
    launch_triton_server
    run_cpp_trtllm_backend_tests
    run_cpp_e2e_backend_tests
    kill_triton_server
    MAX_QUEUE_DELAY_MICROSECONDS="0"

    # -------------------------------
    #  Python BLS
    # -------------------------------

    ACCUMULATE_TOKENS=( "false" "true" )
    E2E_MODEL_NAMES=( "ensemble" "tensorrt_llm_bls" )
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
    E2E_MODEL_NAME="ensemble"
    ACCUMULATE_TOKEN="false"
fi

if [ "$MODEL" = "blip2-opt" ]; then

    MAX_TOKENS_IN_KV_CACHE="${MAX_TOKENS_IN_KV_CACHES[0]}"
    BATCH_SCHEDULER_POLICY="${BATCH_SCHEDULER_POLICIES[0]}"
    KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTIONS[0]}"

    # Test none-streaming
    DECOUPLED_MODE="False"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
        launch_triton_server
        python3 tools/multimodal/blip2_opt2.7b_client.py
        kill_triton_server
    done

    # Test streaming
    DECOUPLED_MODE="True"
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
        launch_triton_server
        python3 tools/multimodal/blip2_opt2.7b_client.py --streaming
        kill_triton_server
    done
    DECOUPLED_MODE="False"

    # Python BLS
    DECOUPLED_MODE="True"
    ACCUMULATE_TOKENS=( "false" "true" )
    E2E_MODEL_NAMES=( "ensemble" "tensorrt_llm_bls" )
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for E2E_MODEL_NAME in "${E2E_MODEL_NAMES[@]}"; do
    for ACCUMULATE_TOKEN in "${ACCUMULATE_TOKENS[@]}"; do

        if [[ "${E2E_MODEL_NAME}" == "ensemble" && "${ACCUMULATE_TOKEN}" == "true" ]]; then
            continue
        fi
        launch_triton_server
        python3 tools/multimodal/blip2_opt2.7b_client.py --use_bls --streaming
        kill_triton_server
    done
    done
    done
    E2E_MODEL_NAME="ensemble"
    ACCUMULATE_TOKEN="false"
    DECOUPLED_MODE="False"

fi
