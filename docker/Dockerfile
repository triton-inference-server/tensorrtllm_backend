FROM nvcr.io/nvidia/tritonserver:22.06-py3
LABEL maintainer="NVIDIA CORPORATION"

ENV PATH /usr/local/cuda/bin:/opt/tritonserver/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && apt-get install -y zlib1g-dev sudo lsof

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-Linux-x86_64.sh && \
    chmod +x cmake-3.21.0-Linux-x86_64.sh && \
    ./cmake-3.21.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.21.0-Linux-x86_64.sh

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install --upgrade numpy

# Install pytorch
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all]
RUN pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH="${TRT_LIBPATH}:${LD_LIBRARY_PATH}"

# Set workspace
ENV WORKSPACE /workspace
WORKDIR /workspace
