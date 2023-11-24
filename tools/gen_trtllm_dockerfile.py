#!/usr/bin/env python3
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

import argparse

FLAGS = None


def install_new_version_of_TRT(clone_repo=False, trtllm_be_repo_tag="main"):
    df = """
# Remove previous TRT installation
RUN apt-get remove --purge -y tensorrt* libnvinfer*
RUN pip uninstall -y tensorrt

# Install new version of TRT using the script from TRT-LLM
RUN apt-get update && apt-get install -y --no-install-recommends python-is-python3
"""
    if clone_repo:
        df += """
RUN git clone --single-branch --depth=1 -b {} https://github.com/triton-inference-server/tensorrtllm_backend.git
RUN cd tensorrtllm_backend && git submodule update --init --recursive
RUN cp tensorrtllm_backend/tensorrt_llm/docker/common/install_tensorrt.sh /tmp/
RUN rm -fr tensorrtllm_backend
    """.format(trtllm_be_repo_tag)
    else:
        df += """
    COPY tensorrt_llm/docker/common/install_tensorrt.sh /tmp/
    """
    df += """
ENV TRT_VER=9.1.0.4
ENV CUDA_VER=12.2
ENV CUDNN_VER=8.9.4.25-1+cuda12.2
ENV NCCL_VER=2.18.3-1+cuda12.2
ENV CUBLAS_VER=12.2.5.6-1
RUN bash /tmp/install_tensorrt.sh && rm /tmp/install_tensorrt.sh

ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH
ENV TRT_ROOT=/usr/local/tensorrt
    """
    return df


def install_pytorch():
    df = """
# Install PyTorch using the script from TRT-LLM
COPY tensorrt_llm/docker/common/install_pytorch.sh /tmp/
# `pypi` for x86_64 arch and `src_cxx11_abi` for aarch64 arch
RUN ARCH="$(uname -i)" && \
    if [ "${ARCH}" = "aarch64" ]; then TORCH_INSTALL_TYPE="src_non_cxx11_abi"; \
    else TORCH_INSTALL_TYPE="pypi"; fi && \
    bash /tmp/install_pytorch.sh $TORCH_INSTALL_TYPE && rm /tmp/install_pytorch.sh
"""

    return df


def create_postbuild(repo_tag="main"):
    df = """
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip
"""
    df += install_new_version_of_TRT(clone_repo=True,
                                     trtllm_be_repo_tag=repo_tag)
    df += """
# Remove TRT contents that are not needed in runtime
RUN ARCH="$(uname -i)" && \
    rm -fr ${TRT_ROOT}/bin ${TRT_ROOT}/targets/${ARCH}-linux-gnu/bin ${TRT_ROOT}/data && \
    rm -fr  ${TRT_ROOT}/doc ${TRT_ROOT}/onnx_graphsurgeon ${TRT_ROOT}/python && \
    rm -fr ${TRT_ROOT}/samples  ${TRT_ROOT}/targets/${ARCH}-linux-gnu/samples

# Install required packages for TRT-LLM models
RUN python3 -m pip install --upgrade pip && \
        pip3 install transformers

# Uninstall unused nvidia packages
RUN if pip freeze | grep -q "nvidia.*"; then \
        pip freeze | grep "nvidia.*" | xargs pip uninstall -y; \
    fi
RUN pip cache purge

ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:$LD_LIBRARY_PATH
"""
    return df


def dockerfile_for_linux():
    df = """
ARG BASE_IMAGE={}
""".format(FLAGS.trtllm_base_image)
    df += """
FROM ${BASE_IMAGE} as base
WORKDIR /workspace
RUN apt-get update && apt-get install python3-pip -y
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

FROM base as dev

# CMake
COPY tensorrt_llm/docker/common/install_cmake.sh /tmp/
RUN bash /tmp/install_cmake.sh && rm /tmp/install_cmake.sh
ENV PATH="/usr/local/cmake/bin:${PATH}"
"""
    df += install_new_version_of_TRT()
    df += install_pytorch()
    df += """

COPY tensorrt_llm/requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com

FROM dev as trt_llm_builder

WORKDIR /app
COPY scripts scripts
COPY tensorrt_llm tensorrt_llm
"""
    df += """
ARG TRTLLM_BUILD_CONFIG={}
""".format(FLAGS.trtllm_build_config)
    df += """
RUN cd tensorrt_llm && python3 scripts/build_wheel.py --build_type=${TRTLLM_BUILD_CONFIG} --trt_root="${TRT_ROOT}" -i --clean
"""

    if FLAGS.trtllm_build_config == "Release":
        df += """
ARG TRTLLM_BUILD_LIB=tensorrt_llm/cpp/build/tensorrt_llm
"""
    else:
        df += """
ARG TRTLLM_BUILD_LIB=tensorrt_llm/cpp/build_{}/tensorrt_llm
""".format(FLAGS.trtllm_build_config)

    df += """
# Copy all artifacts needed by the backend to /opt/trtllm_lib
RUN mkdir -p /opt/trtllm_lib && \
    cp ${TRTLLM_BUILD_LIB}/plugins/libnvinfer_plugin_tensorrt_llm.so /opt/trtllm_lib && \
    (cd /opt/trtllm_lib && \
        ln -s libnvinfer_plugin_tensorrt_llm.so libnvinfer_plugin_tensorrt_llm.so.9 && \
        ln -s libnvinfer_plugin_tensorrt_llm.so libnvinfer_plugin_tensorrt_llm.so.9.1.0)
"""

    with open(FLAGS.output, "w") as dfile:
        dfile.write(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trtllm-base-image",
        type=str,
        required=True,
        help="Base image for building TRT-LLM.",
    )
    parser.add_argument(
        "--trtllm-build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
        help="TRT-LLM build configuration.",
    )
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="File to write Dockerfile to.")

    FLAGS = parser.parse_args()

    dockerfile_for_linux()
