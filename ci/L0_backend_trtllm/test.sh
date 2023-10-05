#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#TENSORRTLLM_BRANCH_TAG=${TENSORRTLLM_BRANCH_TAG:="main"}
#TENSORRTLLM_BRANCH=${TENSORRTLLM_BRANCH:="https://github.com/triton-inference-server/tensorrtllm_backend.git"}
SERVER_IPADDR=${TRITONSERVER_IPADDR:=localhost}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
SERVER_LOG="$PWD/server.log"
CLIENT_LOG="$PWD/client"
DATASET="$PWD/simple_data.json"
TOOLS_DIR='/opt/tritonserver/tensorrtllm_backend/tools'
MODEL_DIR="$PWD/triton_model_repo"
SERVER=/opt/tritonserver/bin/tritonserver
TOKENIZER_DIR=/opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm/tokenizer
BASE_DIR=/opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm
GPT_DIR=/opt/tritonserver/tensorrtllm_backend/tensorrt_llm/examples/gpt
SERVER_PID=0

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
  SERVER_PID=$(pgrep tritonserver)
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local spids="${1}"; shift
    local wait_time_secs="${1:-30}"; shift

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
    pgrep tritonserver | xargs kill -9
}

# =======================================

rm -f $SERVER_LOG* $CLIENT_LOG*
source ./generate_engines.sh
python3 -m pip install --upgrade pip && \
    pip3 install transformers && \
    pip3 install torch && \
    pip3 install tritonclient[all] && \

RET=0

reset_model_repo

# 1-GPU TRT engine 
# inflight batching OFF
# # streaming OFF
cp -r /opt/tritonserver/tensorrtllm_backend/all_models/inflight_batcher_llm/* ${MODEL_DIR}
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags 'inflight_batcher_llm' 'tensorrtllm' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags 'inflight_fused_batching' 'V1' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${engine_dir}' "${BASE_DIR}/engines/inflight_single_gpu/" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/postprocessing/config.pbtxt"
# Copy the engine and place it into the model folder
cp -r ${BASE_DIR}/engines/inflight_single_gpu/ triton_model_repo/tensorrt_llm/1
#docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --runtime=nvidia --shm-size=2g --gpus all -v /home/fpetrini/Desktop/rrtllm/:/workspace tritonserver:w_trt_llm_backend bash
SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/identity_test.py \
    --max_input_len=500 \
    --dataset=${DATASET} \
    --tokenizer_dir=${TOKENIZER_DIR}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching identity test: line ${LINENO}\n***"
    RET=1
fi
set -e

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
    --max_input_len=500 \
    --dataset=${DATASET}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

# 1-GPU TRT engine 
# inflight batching ON
# streaming OFF
replace_config_tags 'V1' 'inflight_fused_batching' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/identity_test.py \
    --max_input_len=500 \
    --dataset=${DATASET} \
    --tokenizer_dir=${TOKENIZER_DIR}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching identity test: line ${LINENO}\n***"
    RET=1
fi
set -e

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
    --max_input_len=500 \
    --dataset=${DATASET}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

# 1-GPU TRT engine 
# inflight batching ON
# streaming ON
replace_config_tags 'decoupled: False' 'decoupled: True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_streaming_client.py \
    --prompt="My name is"

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

# Do not move on to multi-GPU tests
# unless sufficient GPUs exist
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -le 4 ]; then
    exit $RET
fi

# 4-GPU TRT engine 
# inflight batching OFF
# streaming OFF
reset_model_repo

cp -r /opt/tritonserver/tensorrtllm_backend/all_models/inflight_batcher_llm/* ${MODEL_DIR}
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags 'inflight_batcher_llm' 'tensorrtllm' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags 'inflight_fused_batching' 'V1' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${engine_dir}' "${BASE_DIR}/engines/inflight_multi_gpu/" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/postprocessing/config.pbtxt"
# Copy the engine and place it into the model folder
cp -r ${BASE_DIR}/engines/inflight_multi_gpu/ triton_model_repo/tensorrt_llm/1

SERVER_ARGS="--world_size=4 --model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_streaming_client.py \
    --prompt="My name is"

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

# 4-GPU TRT engine 
# inflight batching ON
# streaming OFF
replace_config_tags 'V1' 'inflight_fused_batching' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--world_size=4 --model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_streaming_client.py \
    --prompt="My name is"

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

# 4-GPU TRT engine 
# inflight batching ON
# streaming ON
replace_config_tags 'decoupled: False' 'decoupled: True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--world_size=4 --model_repo=${MODEL_DIR}"
run_server "${SERVER_ARGS}"
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_streaming_client.py \
    --prompt="My name is"

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill_server
sleep 2

exit $RET
