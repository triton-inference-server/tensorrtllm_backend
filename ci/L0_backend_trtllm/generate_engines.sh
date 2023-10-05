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
    # ./c-model/gpt2/ must already exist (it will if build_base_model
    # has already been run)
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

# Install TRT LLM
# FIXME: Update the url
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

# Now that the engines are generated, we should remove the
# tensorrt_llm module to ensure the C++ backend tests are
# not using it
rm -rf /usr/local/lib/python3.10/dist-packages/tensorrt_llm

