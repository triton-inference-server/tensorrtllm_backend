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


def create_postbuild():
    df = """
WORKDIR /workspace

# Remove prevous TRT installation
RUN apt-get remove --purge -y tensorrt* libnvinfer*
RUN pip uninstall -y tensorrt

# Download & install internal TRT release
ARG TENSOR_RT_VERSION="9.1.0.1"
ARG CUDA_VERSION="12.2"
ARG RELEASE_URL_TRT
ARG TARGETARCH=$(uname -i)

RUN --mount=type=cache,target=/root/.cache \
    ARCH=${TARGETARCH} && \
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

# Uninstall unused nvidia packages
RUN if pip freeze | grep -q "nvidia.*"; then \
        pip freeze | grep "nvidia.*" | xargs pip uninstall -y; \
    fi
RUN pip cache purge

ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH
ENV TRT_ROOT=/usr/local/tensorrt
    """
    return df


def dockerfile_for_linux(output_file):
    df = """
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:{}-py3
""".format(FLAGS.triton_container_version)
    df += """
FROM ${BASE_IMAGE} as base
WORKDIR /workspace
"""

    df += """
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

# Remove prevous TRT installation
RUN apt-get remove --purge -y tensorrt* libnvinfer*
RUN pip uninstall -y tensorrt

# Download & install internal TRT release
ARG TENSOR_RT_VERSION="9.1.0.1"
ARG CUDA_VERSION="12.2"
ARG RELEASE_URL_TRT
ARG TARGETARCH=$(uname -i)

RUN --mount=type=cache,target=/root/.cache \
    ARCH=${TARGETARCH} && \
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

ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH
ENV TRT_ROOT=/usr/local/tensorrt

FROM base as dev

# CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1-Linux-x86_64.sh
RUN bash cmake-3.18.1-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir
ENV PATH="/usr/local/bin:${PATH}"

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

    with open(output_file, "w") as dfile:
        dfile.write(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-container-version",
        type=str,
        required=True,
        help="Triton container to use for TRT-LLM build.",
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

    dockerfile_for_linux(FLAGS.output)
