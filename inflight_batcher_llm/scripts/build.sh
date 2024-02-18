#!/bin/bash

TRT_ROOT=${1:-'/usr/local/tensorrt'}

set -x
apt-get update
apt-get install -y --no-install-recommends rapidjson-dev

BUILD_DIR=$(dirname $0)/../build
mkdir $BUILD_DIR
BUILD_DIR=$(cd -- "$BUILD_DIR" && pwd)
cd $BUILD_DIR

export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH}"

cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
