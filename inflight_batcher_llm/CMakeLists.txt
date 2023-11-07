# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of NVIDIA CORPORATION nor the names
# of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
# EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/set_ifndef.cmake)

set(TRITON_BUILD
    OFF
    CACHE STRING "Using Triton build process")

if(TRITON_BUILD)
  set_ifndef(TRTLLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm)
  # Install build time dependencies. This section is executed during cmake
  # configure time.
  execute_process(
    COMMAND bash -x ./tools/environment_setup.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE CMD_RESULT)
  if(NOT CMD_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to install build time dependencies")
  endif()
else()
  set_ifndef(TRTLLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_llm)
endif()

include(${TRTLLM_DIR}/cpp/cmake/modules/find_library_create_target.cmake)

project(tritontensorrtllmbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any projects
# included in this one by FetchContent.
#
# TRITON_ENABLE_GPU is set to OFF as currently the code does not use any GPU
# related features since TRT-LLM backend manages the usage on GPUs itself.
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

# The TRTLLM_BUILD_CONTAINER is used to compile the TRT-LLM libraries that are
# needed for the TRT-LLM backend. The TRTLLM_BUILD_CONTAINER is launched
# separately, and the artifacts will be copied back to the backend installation
# directory.
if(TRITON_BUILD)
  set(TRTLLM_BUILD_CONTAINER
      ""
      CACHE STRING "Base image for building TRT-LLM")
endif()

set(TRITON_COMMON_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all repos that
# we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# Compile TRT-LLM
if(TRITON_BUILD)
  set(TRITON_TRTLLM_DOCKER_NAME "tritonserver-trtllm")
  add_custom_command(
    OUTPUT tensorrt_llm_build
    COMMENT "Building TensorRT-LLM"
    COMMAND
      cd ${CMAKE_CURRENT_SOURCE_DIR} && python3 tools/gen_trtllm_dockerfile.py
      --trtllm-build-config="${CMAKE_BUILD_TYPE}"
      --trtllm-base-image="${TRTLLM_BUILD_CONTAINER}" --output=Dockerfile.trtllm
    COMMAND
      cd ${CMAKE_CURRENT_SOURCE_DIR} && docker build
      --cache-from=${TRITON_TRTLLM_DOCKER_NAME}
      --cache-from=${TRITON_TRTLLM_DOCKER_NAME}_cache0
      --cache-from=${TRITON_TRTLLM_DOCKER_NAME}_cache1 -t
      ${TRITON_TRTLLM_DOCKER_NAME} -f ./Dockerfile.trtllm .
    COMMAND docker rm trtllm_build || echo 'error ignored...' || true
    COMMAND docker create --name trtllm_build ${TRITON_TRTLLM_DOCKER_NAME}
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && rm -fr tensorrt_llm
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && docker cp
            trtllm_build:/app/tensorrt_llm tensorrt_llm
    COMMAND docker cp trtllm_build:/opt/trtllm_lib trtllm_build
    COMMAND docker rm trtllm_build)
endif()

#
# The backend must be built into a shared library. Use an ldscript to hide all
# symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_tensorrtllm.ldscript
               libtriton_tensorrtllm.ldscript COPYONLY)
add_library(triton-tensorrt-llm-backend SHARED src/libtensorrtllm.cc)

if(TRITON_BUILD)
  add_custom_target(trtllm_target DEPENDS tensorrt_llm_build)
  add_dependencies(triton-tensorrt-llm-backend trtllm_target)
endif()

add_library(TritonTensorRTLLMBackend::triton-tensorrt-llm-backend ALIAS
            triton-tensorrt-llm-backend)

enable_language(CUDA)

find_package(CUDA ${CUDA_REQUIRED_VERSION} REQUIRED)

find_library(
  CUDNN_LIB cudnn
  HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR}
  PATH_SUFFIXES lib64 lib)
find_library(
  CUBLAS_LIB cublas
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUBLASLT_LIB cublasLt
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
find_library(
  CUDA_DRV_LIB cuda
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
set(CUDA_LIBRARIES ${CUDART_LIB})

find_package(MPI REQUIRED)
message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_LIBRARIES: ${MPI_LIBRARIES}")

# NCCL dependencies
set_ifndef(NCCL_LIB_DIR /usr/lib/x86_64-linux-gnu/)
set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})

# TRT_LIB_DIR and TRT_INCLUDE_DIR should be aligned with the path in the
# environment_setup.sh script
set_ifndef(TRT_LIB_DIR
           /usr/local/tensorrt/targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/lib)
set_ifndef(
  TRT_INCLUDE_DIR
  /usr/local/tensorrt/targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/include)

set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})

file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" VERSION_STRINGS
     REGEX "#define NV_TENSORRT_.*")
foreach(TYPE MAJOR MINOR PATCH BUILD)
  string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING
               ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

foreach(TYPE MAJOR MINOR PATCH)
  string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING
               ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION
    "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}"
    CACHE STRING "TensorRT project version")
set(TRT_SOVERSION
    "${TRT_SO_MAJOR}"
    CACHE STRING "TensorRT library so version")
message(
  STATUS
    "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}"
)

list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
include_directories(${COMMON_HEADER_DIRS})

target_include_directories(
  triton-tensorrt-llm-backend
  PRIVATE ${TRTLLM_DIR}/cpp
          ${TRTLLM_DIR}/cpp/include
          ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CUDA_INCLUDE_DIRS}
          ${CUDNN_ROOT_DIR}/include
          ${NCCL_INCLUDE_DIR}
          ${3RDPARTY_DIR}/cutlass/include
          ${MPI_INCLUDE_PATH}
          ${COMMON_HEADER_DIR})

target_compile_features(triton-tensorrt-llm-backend PRIVATE cxx_std_17)
target_compile_options(
  triton-tensorrt-llm-backend
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-type-limits>
    $<$<CXX_COMPILER_ID:MSVC>:/Wall
    /D_WIN32_WINNT=0x0A00
    /EHsc>)

add_library(tensorrt_llm STATIC IMPORTED)
set_property(
  TARGET tensorrt_llm
  PROPERTY IMPORTED_LOCATION
           "${TRTLLM_DIR}/cpp/build/tensorrt_llm/libtensorrt_llm_static.a")

add_library(tensorrt_llm_batch_manager STATIC IMPORTED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c"
          "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)
message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")

set(BATCH_MANAGER_ARCH "unknown")

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(BATCH_MANAGER_ARCH "x86_64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(BATCH_MANAGER_ARCH "aarch64-linux-gnu")
else()
  message(
    FATAL_ERROR
      "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(USE_CXX11_ABI)
  set_property(
    TARGET tensorrt_llm_batch_manager
    PROPERTY
      IMPORTED_LOCATION
      "${TRTLLM_DIR}/cpp/tensorrt_llm/batch_manager/${BATCH_MANAGER_ARCH}/libtensorrt_llm_batch_manager_static.a"
  )
else()
  set_property(
    TARGET tensorrt_llm_batch_manager
    PROPERTY
      IMPORTED_LOCATION
      "${TRTLLM_DIR}/cpp/tensorrt_llm/batch_manager/${BATCH_MANAGER_ARCH}/libtensorrt_llm_batch_manager_static.pre_cxx11.a"
  )
endif()

add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
  TARGET nvinfer_plugin_tensorrt_llm
  PROPERTY
    IMPORTED_LOCATION
    "${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)

if(TRITON_BUILD)
  add_dependencies(tensorrt_llm trtllm_target)
  add_dependencies(tensorrt_llm_batch_manager trtllm_target)
  add_dependencies(nvinfer_plugin_tensorrt_llm trtllm_target)
endif()

target_link_libraries(
  triton-tensorrt-llm-backend
  PRIVATE tensorrt_llm_batch_manager
          tensorrt_llm
          triton-core-serverapi # from repo-core
          triton-core-backendapi # from repo-core
          triton-core-serverstub # from repo-core
          triton-backend-utils # from repo-backend
          ${MPI_LIBRARIES}
          ${CUDA_LIBRARIES}
          nvinfer
          nvinfer_plugin_tensorrt_llm)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2)

FetchContent_MakeAvailable(json)

target_link_libraries(triton-tensorrt-llm-backend
                      PRIVATE nlohmann_json::nlohmann_json)

if(WIN32)
  set_target_properties(
    triton-tensorrt-llm-backend PROPERTIES POSITION_INDEPENDENT_CODE ON
                                           OUTPUT_NAME triton_tensorrtllm)
else()
  set_target_properties(
    triton-tensorrt-llm-backend
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               OUTPUT_NAME triton_tensorrtllm
               LINK_DEPENDS
               ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorrtllm.ldscript
               LINK_FLAGS "-Wl,--version-script libtriton_tensorrtllm.ldscript")
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonTensorRTLLMBackend)

install(
  TARGETS triton-tensorrt-llm-backend
  EXPORT triton-tensorrt-llm-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tensorrtllm
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tensorrtllm)

if(TRITON_BUILD)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/trtllm_build/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tensorrtllm)
endif()

install(
  EXPORT triton-tensorrt-llm-backend-targets
  FILE TritonTensorRTLLMBackendTargets.cmake
  NAMESPACE TritonTensorRTLLMBackend::
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonTensorRTLLMBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTLLMBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTLLMBackendConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

#
# Export from build tree
#
export(
  EXPORT triton-tensorrt-llm-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTLLMBackendTargets.cmake
  NAMESPACE TritonTensorRTLLMBackend::)

export(PACKAGE TritonTensorRTLLMBackend)
