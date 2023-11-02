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

apt-get update && apt-get install git-lfs rapidjson-dev python3-pip python-is-python3 -y --no-install-recommends
# Update submodule
git lfs install
git submodule update --init --recursive
(cd tensorrt_llm/cpp/tensorrt_llm/batch_manager && git lfs pull)

pip3 install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

# Remove previous TRT installation
apt-get remove --purge -y tensorrt* libnvinfer*
pip uninstall -y tensorrt

# Download & install internal TRT release
bash tensorrt_llm/docker/common/install_tensorrt.sh

# Install PyTorch
ARCH="$(uname -i)"
if [ "$ARCH" = "aarch64" ];then TORCH_INSTALL_TYPE="src_non_cxx11_abi"; else TORCH_INSTALL_TYPE="pypi";fi
bash tensorrt_llm/docker/common/install_pytorch.sh $TORCH_INSTALL_TYPE

export LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH
export TRT_ROOT=/usr/local/tensorrt
