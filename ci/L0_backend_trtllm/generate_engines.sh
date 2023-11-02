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
    local NUM_GPUS=$1
    cd ${GPT_DIR}
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
    python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism ${NUM_GPUS} --storage-type float16
    cd ${BASE_DIR}
}

function build_tensorrt_engine_inflight_batcher {
    local NUM_GPUS=$1
    cd ${GPT_DIR}
    local GPT_MODEL_DIR=./c-model/gpt2/${NUM_GPUS}-gpu/
    local OUTPUT_DIR=inflight_${NUM_GPUS}_gpu/
    # ./c-model/gpt2/ must already exist (it will if build_base_model
    # has already been run)
    python3 build.py --model_dir="${GPT_MODEL_DIR}" \
                 --world_size="${NUM_GPUS}" \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir="${OUTPUT_DIR}"
    cd ${BASE_DIR}
}

function install_trt_llm {
    # Install CMake
    bash /opt/tritonserver/tensorrtllm_backend/tensorrt_llm/docker/common/install_cmake.sh
    export PATH="/usr/local/cmake/bin:${PATH}"

    # PyTorch needs to be built from source for aarch64
    ARCH="$(uname -i)"
    if [ "${ARCH}" = "aarch64" ]; then TORCH_INSTALL_TYPE="src_non_cxx11_abi"; \
    else TORCH_INSTALL_TYPE="pypi"; fi && \
    (cd /opt/tritonserver/tensorrtllm_backend/tensorrt_llm &&
        bash docker/common/install_pytorch.sh $TORCH_INSTALL_TYPE &&
        python3 ./scripts/build_wheel.py --trt_root="${TRT_ROOT}" &&
        pip3 install ./build/tensorrt_llm*.whl)
}

# Install TRT LLM
install_trt_llm

# Generate the TRT_LLM model engines
NUM_GPUS_TO_TEST=("1" "2" "4")
for NUM_GPU in "${NUM_GPUS_TO_TEST[@]}"; do
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPU" ]; then
        continue
    fi

    build_base_model "${NUM_GPU}"
    build_tensorrt_engine_inflight_batcher "${NUM_GPU}"
done

# Move the TRT_LLM model engines to the CI directory
mkdir engines
mv ${GPT_DIR}/inflight_*_gpu/ engines/

# Move the tokenizer into the CI directory
mkdir tokenizer
mv ${GPT_DIR}/gpt2/* tokenizer/

# Now that the engines are generated, we should remove the
# tensorrt_llm module to ensure the C++ backend tests are
# not using it
pip3 uninstall -y torch tensorrt_llm
