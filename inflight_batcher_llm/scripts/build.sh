#!/bin/bash

TRT_ROOT=${1:-'/usr/local/tensorrt'}

set -x
apt-get update
apt-get install -y --no-install-recommends rapidjson-dev

BUILD_DIR=$(dirname $0)/../build
mkdir $BUILD_DIR
BUILD_DIR=$(cd -- "$BUILD_DIR" && pwd)
cd $BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRT_LIB_DIR=${TRT_ROOT}/targets/x86_64-linux-gnu/lib \
    -DTRT_INCLUDE_DIR=${TRT_ROOT}/include ..
make install
