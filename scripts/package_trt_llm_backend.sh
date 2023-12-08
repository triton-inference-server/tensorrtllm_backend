# https://gitlab-master.nvidia.com/ftp/tekit_tool/-/blob/6179b502128dbb453a2fd3911b61beb5f4c7cb40/tar_tensorrt_llm_backend.sh

#!/usr/bin/env bash

set -ex

TGT_PKG_NAME=${1}
BACKEND_DIR=${2}

pushd ${BACKEND_DIR}

# clean files
rm -rf .git
rm .gitmodules
rm -rf tekit
rm -rf tensorrt_llm

rm dockerfile/Dockerfile.dev

rm scripts/package_trt_llm_backend.sh

rm -rf tests

rm -rf tools/dataset

rm -rf jenkins

#
# Closed-source Batch Manager changes
#

# patch the cmake to use lib with pre_cxx11.a suffix and remove the patch from release package
patch -N inflight_batcher_llm/CMakeLists.txt -i inflight_batcher_llm/CMakeLists.txt.patch
rm inflight_batcher_llm/CMakeLists.txt.patch
[[ -e inflight_batcher_llm/CMakeLists.txt.orig ]] && rm inflight_batcher_llm/CMakeLists.txt.orig

# exit if the keyword is found
grep "__LUNOWUD" -R > /dev/null && exit 1
grep "tekit" -R > /dev/null && exit 1

# package
popd
tar -czvf ${TGT_PKG_NAME} ${BACKEND_DIR}
