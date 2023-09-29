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
SERVER_LOG="$PWD/server_log"
CLIENT_LOG="$PWD/client"
DATASET="$PWD/simple_data.json"
TOOLS_DIR="$PWD/tensorrtllm_backend/tools/"
MODEL_DIR="$PWD/triton_model_repo"
SERVER=/opt/tritonserver/bin/tritonserver
TOKENIZER_DIR=/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
SERVER_PID=0

#MODEL_DIR=${MODEL_DIR:=$PWD/tensorrtllm_backend/all_models/gpt/}
#TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
#SERVER=${TRITON_DIR}/bin/tritonserver
#BACKEND_DIR=${TRITON_DIR}/backends
#SERVER_ARGS_EXTRA="--exit-timeout-secs=${SERVER_TIMEOUT} --backend-directory=${BACKEND_DIR}"
#SERVER_ARGS="--model-repository=${MODEL_DIR} ${SERVER_ARGS_EXTRA}"
#source ../common/util.sh

# Helpers ===============================
function replace_config_tags {
  tag_to_replace="${1}"
  new_value="${2}"
  config_file_path="${3}"
  sed -i "s|${tag_to_replace}|${new_value}|g" ${config_file_path}
  
}

function build_tensorrt_engine_gpt {
    cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
    python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
    python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin
    #TODO find out where top-level directory is
    cd /workspace
}

function build_tensorrt_engine_inflight_batcher {
    cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt
    # ./c-model/gpt2/ must already exist (it will if build_tensorrt_engine_gpt)
    # has already been run
    python3 build.py --model_dir=./c-model/gpt2/1-gpu/ \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --output_dir=engines/fp16/1-gpu
    cd /workspace

}

function build_tensorrt_engine_inflight_batcher_multi_gpu {
    cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt
    python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism --tensor-parallelism 4 --storage-type float16
    python3 build.py --model_dir=./c-model/gpt2/4-gpu/ \
                 --world_size=4 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --output_dir=gpt_multi_gpu
    cd /workspace
}

function run_server {
  SERVER_ARGS="${1}"
  python3 /app/scripts/launch_triton_server.py ${SERVER_ARGS} > ${SERVER_LOG} 2>&1 &
  sleep 2 # allow time to obtain the pid
  SERVER_PID=$(pgrep tritonserver)
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local spid="${1}"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid > /dev/null 2>&1; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

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

# =======================================

rm -f $SERVER_LOG* $CLIENT_LOG*


RET=0

#git clone --single-branch --depth=1 -b ${TENSORRTLLM_BRANCH_TAG} ${TENSORRTLLM_BRANCH}
#git clone --single-branch --depth=1 -b "main" git@github.com:triton-inference-server/tensorrtllm_backend.git
# docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --runtime=nvidia --gpus all --shm-size=2g -v /home/fpetrini/Desktop/rrtllm/:/workspace tritonserver:w_trt_llm_backend bash
# docker run -it --rm --runtime=nvidia --gpus all --shm-size=2g -v /home/fpetrini/trt_llm/:/workspace 3e96065a3dcc bash

# 1-GPU TRT engine with gpt
reset_model_repo
build_tensorrt_engine_gpt
cp -r /workspace/tensorrtllm_backend/all_models/gpt/* ${MODEL_DIR}
replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${engine_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt_outputs/' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' "${MODEL_DIR}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/postprocessing/config.pbtxt"

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server $SERVER_ARGS
wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
if [ "$WAIT_RET" != "0" ]; then
    # Cleanup
    kill $SERVER_PID > /dev/null 2>&1 || true
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 ${TOOLS_DIR}/gpt/client.py \
    --text="Born in north-east France, Soyer trained as a" \
    --output_len=10 \
    --tokenizer_dir gpt2 \
    --tokenizer_type auto

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing gpt client: line ${LINENO}\n***"
    RET=1
fi
set -e

set +e
python3 ${TOOLS_DIR}/gpt/identity_test.py \
    --batch_size=8 \
    --start_len=128 \
    --output_len=20

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing gpt identity test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill -9 $SERVER_PID
sleep 2

# 1-GPU TRT engine with inflight_batcher_llm 
# inflight batching OFF
# # streaming OFF
reset_model_repo
build_tensorrt_engine_inflight_batcher

cp -r /workspace/tensorrtllm_backend/all_models/inflight_batcher_llm/* ${MODEL_DIR}
replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/preprocessing/config.pbtxt"
replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags 'inflight_fused_batching' 'V1' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${engine_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' "${MODEL_DIR}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_DIR}/postprocessing/config.pbtxt"
# Copy the engine and place it into the model folder
cp -r /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/ triton_model_repo/tensorrt_llm/1

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server $SERVER_ARGS
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
    --max_input_len=100 \
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
    --max_input_len=100 \
    --dataset=${DATASET}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill -9 $SERVER_PID
sleep 2

replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

# 1-GPU TRT engine with inflight_batcher_llm 
# inflight batching ON
# streaming OFF
replace_config_tags 'V1' 'inflight_fused_batching' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server $SERVER_ARGS
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
    --max_input_len=100 \
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
    --max_input_len=100 \
    --dataset=${DATASET}

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Error executing inflight batching end-to-end test: line ${LINENO}\n***"
    RET=1
fi
set -e

kill -9 $SERVER_PID
sleep 2

# 1-GPU TRT engine with inflight_batcher_llm 
# inflight batching ON
# streaming ON
replace_config_tags '${decoupled_mode}' 'True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

SERVER_ARGS="--model_repo=${MODEL_DIR}"
run_server $SERVER_ARGS
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

kill -9 $SERVER_PID
sleep 2

exit 1
# 4-GPU TRT engine with inflight_batcher_llm 
# inflight batching OFF
# streaming OFF
reset_model_repo
build_tensorrt_engine_inflight_batcher_multi_gpu


#cp -r all_models/inflight_batcher_llm/* triton_model_repo
#replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' 'triton_model_repo/preprocessing/config.pbtxt'
#replace_config_tags '${tokenizer_type}' 'auto' 'triton_model_repo/preprocessing/config.pbtxt'
#replace_config_tags '${engine_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt_outputs/' 'triton_model_repo/tensorrt_llm/config.pbtxt'
#replace_config_tags '${tokenizer_dir}' '/workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2/' 'triton_model_repo/postprocessing/config.pbtxt'
#replace_config_tags '${tokenizer_type}' 'auto' 'triton_model_repo/postprocessing/config.pbtxt'


#docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --runtime=nvidia --gpus all --shm-size=2g -v /home/fpetrini/Desktop/rrtllm/tensorrtllm_backend:/workspace c4e5d98f640a bash


# do not need to copy over model.py for all_models/gpt
# engine path for non in-flight only
#sed -i "s|\${engine_dir}|/workspace/tensorrt_llm/examples/gpt/gpt_outputs/|g" triton_model_repo/tensorrt_llm/config.pbtxt
#sed -i "s|\${tokenizer_dir}|tensorrt_llm/examples/gpt/gpt2|g" triton_model_repo/preprocessing/config.pbtxt
#sed -i "s|\${tokenizer_type}|auto|g" triton_model_repo/preprocessing/config.pbtxt

# docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --runtime=nvidia --gpus all --shm-size=2g -v ${PWD}:/workspace c4e5d98f640a bash
