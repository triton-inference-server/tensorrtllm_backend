#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
SERVER_IPADDR=${TRITONSERVER_IPADDR:=localhost}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
DATASET="$PWD/simple_data.json"
TOOLS_DIR='/opt/tritonserver/tensorrtllm_backend/tools'
STREAM_DIR='/opt/tritonserver/tensorrtllm_backend/inflight_batcher_llm/client'
MODEL_DIR="$PWD/triton_model_repo"
SERVER=/opt/tritonserver/bin/tritonserver
TOKENIZER_DIR=/opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm/tokenizer
BASE_DIR=/opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm
BASE_METRICS_VERIFICATION_TEST=base_metrics_verification_tests.py
BASE_METRICS_VERIFICATION_LOG="base_metrics_verification.log"
CUSTOM_METRICS_VERIFICATION_TEST=custom_metrics_verification_tests.py
CUSTOM_METRICS_VERIFICATION_LOG="custom_metrics_verification.log"
SERVER_PID=0

# Force environment to use python version 3
apt update -q=2 \
    && apt install -y python-is-python3

# Helpers ===============================
function replace_config_tags {
  tag_to_replace="${1}"
  new_value="${2}"
  config_file_path="${3}"
  sed -i "s|${tag_to_replace}|${new_value}|g" ${config_file_path}

}

function run_server {
  SERVER_ARGS="${1}"
  python3 /opt/tritonserver/tensorrtllm_backend/scripts/launch_triton_server.py ${SERVER_ARGS} > ${SERVER_LOG} 2>&1 &
  sleep 2 # allow time to obtain the pid(s)
  # Read PIDs into an array, trimming whitespaces
  readarray -t SERVER_PID < <(pgrep "tritonserver")
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local wait_time_secs="${1:-30}"; shift
    local spids=("$@");

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        # Multi-GPU will spawn multiple pids
        for pid in "${spids[@]}"; do
            if ! kill -0 $pid > /dev/null 2>&1; then
                echo "=== Server not running."
                WAIT_RET=1
                return
            fi
        done

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} ${SERVER_IPADDR}:8000/v2/health/ready`
        set -e
        if [ "$code" == "200" ]; then
            code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":1}' localhost:8000/v2/logging`
            assert_curl_success "Failed to change log settings necessary for verification" ${BASH_LINENO}
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

function reset_model_repo {
    rm -rf triton_model_repo/
    mkdir ${MODEL_DIR}
}

function kill_server {
    pgrep tritonserver | xargs kill -SIGINT
}

function wait_for_server_terminated {
    local spids=("$@");
    for pid in "${spids[@]}"; do
        echo "Waiting for proc ${pid} to terminate..."
        while true; do
            if ! (kill -0 $pid) > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done
    done
}

function assert_curl_success {
  message="${1}"
  original_line_no="${2}"
  RET=0
  if [ "$code" != "200" ]; then
    cat ./curl.out
    cat ${SERVER_LOG}
    echo -e "\n***\n*** ${message} : line ${original_line_no}\n***"
    RET=1
  fi
  return ${RET}
}

# =======================================

rm -f *.log *.out
# Generate TRT_LLM engines and install dependencies
source ./generate_engines.sh
python3 -m pip install --upgrade pip && \
    pip3 install tritonclient[all] && \
    pip3 install pandas && \
    pip3 install tabulate

RET=0

NUM_GPUS_TO_TEST=("1" "2" "4")
for NUM_GPU in "${NUM_GPUS_TO_TEST[@]}"; do
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPU" ]; then
        exit $RET
    fi

    SERVER_ARGS="--world_size=${NUM_GPU} --model_repo=${MODEL_DIR}"

    reset_model_repo

    cp -r /opt/tritonserver/tensorrtllm_backend/all_models/inflight_batcher_llm/* ${MODEL_DIR}
    rm -rf ${MODEL_DIR}/tensorrt_llm_bls
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/ensemble/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${preprocessing_instance_count}' '1' "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${batching_strategy}' 'INVALID' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${engine_dir}' "${MODEL_DIR}/tensorrt_llm/1/inflight_${NUM_GPU}_gpu/" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' "50000" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/postprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/postprocessing/config.pbtxt"
    replace_config_tags '${postprocessing_instance_count}' '1' "${MODEL_DIR}/postprocessing/config.pbtxt"

    # Copy the engine and place it into the model folder
    cp -r ${BASE_DIR}/engines/inflight_${NUM_GPU}_gpu/ triton_model_repo/tensorrt_llm/1

    # Invalid GPT model Type
    SERVER_LOG="./${NUM_GPU}gpu_invalid_batch_strat.log"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # Expect invalid GPT model type error to be gracefully handled
    if [ `grep -c "Invalid gpt_model_type" $SERVER_LOG` == "0" ]; then
        echo -e "\n***\n*** GPT model type error not handled gracefully: line ${LINENO}\n***"
        cat $SERVER_LOG
        exit 1
    fi

    wait_for_server_terminated ${SERVER_PID[@]}

    # inflight batching OFF (V1)
    # streaming OFF
    SERVER_LOG="./${NUM_GPU}gpu_v1_no_streaming_server.log"
    replace_config_tags 'INVALID' 'V1' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/benchmark_core_model.py \
        --max-input-len=500 \
        dataset --dataset=${DATASET} \
        --tokenizer-dir=${TOKENIZER_DIR}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing v1 benchmark_core_model test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
        --max-input-len=500 \
        --dataset=${DATASET}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing v1 end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    curl localhost:8002/metrics -o ${NUM_GPU}gpu_v1_no_stream_metrics.out
    kill_server
    wait_for_server_terminated ${SERVER_PID[@]}

    # inflight batching ON
    # streaming OFF
    SERVER_LOG="./${NUM_GPU}gpu_IFB_no_streaming_server.log"
    replace_config_tags 'V1' 'inflight_fused_batching' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/benchmark_core_model.py \
        --max-input-len=500 \
        dataset --dataset=${DATASET} \
        --tokenizer-dir=${TOKENIZER_DIR}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching benchmark_core_model test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
        --max-input-len=500 \
        --dataset=${DATASET}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    curl localhost:8002/metrics -o ${NUM_GPU}gpu_IFB_no_stream_metrics.out
    kill_server
    wait_for_server_terminated ${SERVER_PID[@]}

    # Start a clean server to verify base metrics are being
    # reported correctly
    SERVER_LOG="./${NUM_GPU}gpu_IFB_no_streaming_base_metrics.log"
    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    set -e

    python3 ${BASE_METRICS_VERIFICATION_TEST} >> ${BASE_METRICS_VERIFICATION_LOG} 2>&1
    if [ $? -ne 0 ]; then
        cat ${BASE_METRICS_VERIFICATION_LOG}
        RET=1
    fi
    set +e

    kill_server
    wait_for_server_terminated ${SERVER_PID[@]}

    # World size must be 1 when using multi-model
    if [ "${NUM_GPU}" == "0" ]; then
        # Multi-model
        SERVER_LOG="./${NUM_GPU}gpu_multi_model.log"
        run_server "${SERVER_ARGS} --multi-model"
        wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        if [ "$WAIT_RET" != "0" ]; then
            # Cleanup
            kill $SERVER_PID > /dev/null 2>&1 || true
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
        set -e

        python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
            --max-input-len=500 \
            --dataset=${DATASET}

        if [ $? -ne 0 ]; then
            cat $SERVER_LOG
            echo -e "\n***\n*** Error executing inflight batching end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
            kill_server
            wait_for_server_terminated ${SERVER_PID[@]}
            RET=1
        fi
        set +e

        curl localhost:8002/metrics -o ${NUM_GPU}gpu_multi_model_metrics.out
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
    fi

    # inflight batching ON
    # streaming ON
    SERVER_LOG="./${NUM_GPU}gpu_IFB_streaming_server.log"
    replace_config_tags 'decoupled: False' 'decoupled: True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set -e
    python3 ${STREAM_DIR}/end_to_end_grpc_client.py \
        --prompt="My name is"

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    curl localhost:8002/metrics -o ${NUM_GPU}gpu_IFB_stream_metrics.out
    kill_server
    wait_for_server_terminated ${SERVER_PID[@]}

done

# Verify TRT LLM statistics are being properly reported as custom metrics
python3 ${CUSTOM_METRICS_VERIFICATION_TEST} >> ${CUSTOM_METRICS_VERIFICATION_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${CUSTOM_METRICS_VERIFICATION_LOG}
    RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
