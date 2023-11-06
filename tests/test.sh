#!/usr/bin/bash

set -x

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

MAX_NUM_SEQUENCES=( "" "4" "32" )
MAX_TOKENS_IN_KV_CACHES=( "" "2048" )
BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" "max_utilization")
KV_CACHE_FREE_GPU_MEM_FRACTIONS=( "" "0.2" )
ENABLE_TRT_OVERLAPS=( "false" "true" )
BATCHING_STRATEGIES=( "inflight_fused_batching" )
EXCLUDE_INPUT_IN_OUTPUTS=( "true" "false" )

if [ "$MODEL" = "gpt-ib" ]; then
    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    test_count=0
    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do
    for EXCLUDE_INPUT_IN_OUTPUT in "${EXCLUDE_INPUT_IN_OUTPUTS[@]}"; do

        # Because the runners are shared, the default value of 0.85 doesn't work, so skip
        # if max_tokens_in_kv_cache is also empty
        if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
            continue
        fi

        echo "----------------------------------"
        echo "MAX_NUM_SEQUENCES: ${MAX_NUM_SEQUENCE}"
        echo "MAX_TOKENS_IN_KV_CACHE: ${MAX_TOKENS_IN_KV_CACHE}"
        echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
        echo "KV_CACHE_FREE_GPU_MEM_FRACTION: ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
        echo "ENABLE_TRT_OVERLAP: ${ENABLE_TRT_OVERLAP}"
        echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
        echo "----------------------------------"
        rm -rf ./triton_repo
        cp -R all_models/inflight_batcher_llm triton_repo
        # Modify config.pbtxt
        python3 tools/fill_template.py -i triton_repo/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:False,max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},max_num_sequences:${MAX_NUM_SEQUENCE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:${ENABLE_TRT_OVERLAP},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT}
        python3 tools/fill_template.py -i triton_repo/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
        python3 tools/fill_template.py -i triton_repo/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

        # Launch Triton Server
        /opt/tritonserver/bin/tritonserver \
            --model-repository=triton_repo &
        export SERVER_PID=$!
        wait_for_server_ready ${SERVER_PID} 1200

        # Test client
        pushd inflight_batcher_llm/client

        # test to run for all combinations of flags
        EXCL_INPUT_IN_OUTPUT_FLAG=""
        [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"

        python3 inflight_batcher_llm_client.py \
            --check-output \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --tokenizer_dir ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE}


        if [[ "$test_count" == "0" ]]; then

            # test without stop words
            python3 end_to_end_grpc_client.py \
                -o 10 \
                -p "The only thing we have to fear is"  \
                 &> output_wo_stop_words
            cat output_wo_stop_words
            grep "that the government will" output_wo_stop_words
            echo "grep exit code: $?"

            # test with stop words
            python3 end_to_end_grpc_client.py \
                -o 10 \
                -p "The only thing we have to fear is"  \
                --stop_words " government" \
                &> output_w_stop_words
            cat output_w_stop_words
            grep -v "that the government will" output_w_stop_words
            echo "grep exit code: $?"

            # test with request cancellation
            python3 inflight_batcher_llm_client.py \
                --request-output-len=128 \
                --stop-after-ms 100 \
                --tokenizer_dir ${TOKENIZER_PATH} \
                --tokenizer_type ${TOKENIZER_TYPE} &> output_w_stop

            cat output_w_stop
            grep "Got cancellation response" output_w_stop
        fi

        popd # inflight_batcher_llm/client

        # End to end test
        pushd tools/inflight_batcher_llm

        python3 end_to_end_test.py \
            --concurrency 8 \
            -i http \
            --max_input_len 300 \
            --dataset ../dataset/mini_cnn_eval.json

        if [[ "$test_count" == "0" ]]; then
            python3 end_to_end_test.py \
                --concurrency 8 \
                -i grpc \
                --max_input_len 300 \
                --dataset ../dataset/mini_cnn_eval.json
        fi

        EXCL_INPUT_IN_OUTPUT_FLAG=""
        [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude_input_in_output"
        python3 identity_test.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --concurrency 8 \
            -i http \
            --max_input_len 300 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer_dir ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE}
        python3 identity_test.py \
            --concurrency 8 \
            -i grpc \
            --max_input_len 300 \
            dataset \
            --dataset ../dataset/mini_cnn_eval.json \
            --tokenizer_dir ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE}
        python3 identity_test.py \
            --concurrency 8 \
            -i grpc \
            --max_input_len 300 \
            --request_rate -1 \
            token_norm_dist \
            --input_mean 128 --input_stdev 0 \
            --output_mean 20 --output_stdev 0 \
            --num_requests 100

        popd # tools/inflight_batcher_llm

        kill -9 ${SERVER_PID}

        ((test_count=test_count+1))
    done
    done
    done
    done
    done
    done
    done
fi

if [ "$MODEL" = "gpt-ib-streaming" ]; then
    # To make sure that torch is not a dependency for C++ backend
    pip3 uninstall -y torch

    for MAX_NUM_SEQUENCE in "${MAX_NUM_SEQUENCES[@]}"; do
    for MAX_TOKENS_IN_KV_CACHE in "${MAX_TOKENS_IN_KV_CACHES[@]}"; do
    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do
    for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do
    for KV_CACHE_FREE_GPU_MEM_FRACTION in "${KV_CACHE_FREE_GPU_MEM_FRACTIONS[@]}"; do
    for ENABLE_TRT_OVERLAP in "${ENABLE_TRT_OVERLAPS[@]}"; do
    for EXCLUDE_INPUT_IN_OUTPUT in "${EXCLUDE_INPUT_IN_OUTPUTS[@]}"; do

        # Because the runners are shared, the default value of 0.85 doesn't work, so skip
        # if max_tokens_in_kv_cache is also empty
        if [[ "${KV_CACHE_FREE_GPU_MEM_FRACTION}" == "" && "${MAX_TOKENS_IN_KV_CACHE}" == "" ]]; then
            continue
        fi

        echo "----------------------------------"
        echo "MAX_NUM_SEQUENCES: ${MAX_NUM_SEQUENCE}"
        echo "MAX_TOKENS_IN_KV_CACHE: ${MAX_TOKENS_IN_KV_CACHE}"
        echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
        echo "KV_CACHE_FREE_GPU_MEM_FRACTION: ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
        echo "ENABLE_TRT_OVERLAP: ${ENABLE_TRT_OVERLAP}"
        echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
        echo "----------------------------------"

        rm -rf ./triton_repo
        cp -R all_models/inflight_batcher_llm triton_repo
        # Modify config.pbtxt
        python3 tools/fill_template.py -i triton_repo/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:True,max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},max_num_sequences:${MAX_NUM_SEQUENCE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:${ENABLE_TRT_OVERLAP},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT}
        python3 tools/fill_template.py -i triton_repo/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
        python3 tools/fill_template.py -i triton_repo/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

        # Launch Triton Server
        /opt/tritonserver/bin/tritonserver \
            --model-repository=triton_repo &
        export SERVER_PID=$!
        wait_for_server_ready ${SERVER_PID} 1200

        # Test client
        pushd inflight_batcher_llm/client
        EXCL_INPUT_IN_OUTPUT_FLAG=""
        [ "${EXCLUDE_INPUT_IN_OUTPUT}" = "true" ] && EXCL_INPUT_IN_OUTPUT_FLAG="--exclude-input-in-output"
        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming --check-output \
            --tokenizer_dir ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE}

        python3 inflight_batcher_llm_client.py \
            ${EXCL_INPUT_IN_OUTPUT_FLAG} \
            --streaming \
            --request-output-len=128 \
            --stop-after-ms 100 \
            --tokenizer_dir ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} &> output_w_stop

        cat output_w_stop
        grep "Got cancellation response" output_w_stop


        # End to end test
        python3 end_to_end_grpc_client.py \
            --output_len 10 --prompt "This is a test "
        popd

        kill -9 ${SERVER_PID}

    done
    done
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

    BATCHING_STRATEGIES=( "inflight_fused_batching" "v1")

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
        python3 tools/fill_template.py -i triton_repo/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:False,max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},max_num_sequences:${MAX_NUM_SEQUENCE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:${ENABLE_TRT_OVERLAP}
        python3 tools/fill_template.py -i triton_repo/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
        python3 tools/fill_template.py -i triton_repo/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

        # Launch Triton Server
        /opt/tritonserver/bin/tritonserver \
            --model-repository=triton_repo &
        export SERVER_PID=$!
        wait_for_server_ready ${SERVER_PID} 1200

        # Test client
        pushd inflight_batcher_llm/client

        python3 inflight_batcher_llm_client.py --prompt_embedding_table ../../tensorrt_llm/examples/gpt/email_composition.npy --prompt_task_id 0 --input_tokens_csv ../../tensorrt_llm/examples/gpt/input.csv --output_tokens_csv ../../tensorrt_llm/examples/gpt/output_w_prompt.csv --check-output --request-output-len 8

        python3 inflight_batcher_llm_client.py --input_tokens_csv ../../tensorrt_llm/examples/gpt/input_wo_prompt.csv --output_tokens_csv ../../tensorrt_llm/examples/gpt/output_wo_prompt.csv --check-output --request-output-len 8

        popd # inflight_batcher_llm/client

        kill -9 ${SERVER_PID}

    done
fi
