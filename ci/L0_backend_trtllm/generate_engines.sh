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

BASE_DIR=/opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm
GPT_DIR=/opt/tritonserver/tensorrtllm_backend/tensorrt_llm/examples/gpt

function build_base_model {
    cd ${GPT_DIR}
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
    python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
    cd ${BASE_DIR}
}

function build_tensorrt_engine_inflight_batcher {
    cd ${GPT_DIR}
    # ./c-model/gpt2/ must already exist (it will if build_base_model)
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
                 --output_dir=inflight_single_gpu/
    cd ${BASE_DIR}

}

function build_tensorrt_engine_inflight_batcher_multi_gpu {
    cd ${GPT_DIR}
    python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 4 --storage-type float16
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
                 --output_dir=inflight_multi_gpu/
    cd ${BASE_DIR}
}


apt-get remove --purge -y tensorrt* libnvinfer*
pip uninstall -y tensorrt

# Install TRT version > 9.0
TENSOR_RT_VERSION="9.1.0.1"
CUDA_VERSION="12.2"
ARCH=$(uname -i)
if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi && \
if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi && \
if [ "$ARCH" = "x86_64" ];then DIR_NAME="x64-agnostic"; else DIR_NAME=${ARCH};fi &&\
if [ "$ARCH" = "aarch64" ];then OS1="Ubuntu22_04" && OS2="Ubuntu-22.04"; else OS1="Linux" && OS2="Linux";fi &&\
RELEASE_URL_TRT=http://cuda-repo.nvidia.com/release-candidates/Libraries/TensorRT/v9.1/${TENSOR_RT_VERSION}-b6aa91dc/${CUDA_VERSION}-r535/${OS1}-${DIR_NAME}/tar/TensorRT-${TENSOR_RT_VERSION}.${OS2}.${ARCH}-gnu.cuda-${CUDA_VERSION}.tar.gz && \
wget ${RELEASE_URL_TRT} -O /workspace/TensorRT.tar && \
tar -xf /workspace/TensorRT.tar -C /usr/local/ && \
mv /usr/local/TensorRT-${TENSOR_RT_VERSION} /usr/local/tensorrt && \
pip install /usr/local/tensorrt/python/tensorrt-*-cp310-*.whl && \
rm -rf /workspace/TensorRT.tar
pip install git+https://gitlab-master.nvidia.com/fpetrini/TensorRT-LLM.git

mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

export LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH
export TRT_ROOT=/usr/local/tensorrt

# Generate the TRT_LLM model engines
build_base_model
build_tensorrt_engine_inflight_batcher
build_tensorrt_engine_inflight_batcher_multi_gpu

# Move the TRT_LLM model engines to the CI directory
mkdir engines
mv ${GPT_DIR}/inflight_single_gpu engines/
mv ${GPT_DIR}/inflight_multi_gpu engines/

# Move the tokenizer into the CI directory
mkdir tokenizer
mv ${GPT_DIR}/gpt2/* tokenizer/

# FIXME: Current model in all_models contains a tensorrt_llm module dependency.
# The copy of model.py that overwrites the all_models/model.py inlines the
# dependent function.
cp model.py ../../all_models/inflight_batcher_llm/preprocessing/1/model.py
