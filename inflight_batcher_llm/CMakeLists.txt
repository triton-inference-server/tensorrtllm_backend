# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)
include(${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/cmake/modules/set_ifndef.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/cmake/modules/find_library_create_target.cmake)

project(tutorialinflight_batcher_llmbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because inflight_batcher_llm backend doesn't
# use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_inflight_batcher_llm.ldscript libtriton_inflight_batcher_llm.ldscript COPYONLY)

add_library(
  triton-inflight_batcher_llm-backend SHARED
  src/inflight_batcher_llm.cc
)

add_library(
  InflightBatcherLLM::triton-inflight_batcher_llm-backend ALIAS triton-inflight_batcher_llm-backend
)

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

set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})
find_library_create_target(nvuffparser nvparsers SHARED ${TRT_LIB_DIR})


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
  triton-inflight_batcher_llm-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${3RDPARTY_DIR}/cutlass/include
    ${MPI_INCLUDE_PATH}
    ${COMMON_HEADER_DIR}
)

#include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_ROOT_DIR}/include
#                    ${NCCL_INCLUDE_DIR} ${3RDPARTY_DIR}/cutlass/include)

target_compile_features(triton-inflight_batcher_llm-backend PRIVATE cxx_std_17)
target_compile_options(
  triton-inflight_batcher_llm-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

add_library(tensorrt_llm STATIC IMPORTED)
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/build/tensorrt_llm/libtensorrt_llm_static.a")

add_library(tensorrt_llm_batch_manager STATIC IMPORTED)
set_property(TARGET tensorrt_llm_batch_manager PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/build/tensorrt_llm/batch_manager/libtensorrt_llm_batch_manager_static.a")

add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../tekit/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so")

target_link_libraries(
  triton-inflight_batcher_llm-backend
  PRIVATE
    tensorrt_llm_batch_manager
    tensorrt_llm
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${MPI_LIBRARIES}
    nvinfer
    nvinfer_plugin_tensorrt_llm
)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2)

FetchContent_MakeAvailable(json)

target_link_libraries(triton-inflight_batcher_llm-backend PRIVATE nlohmann_json::nlohmann_json)

if(WIN32)
  set_target_properties(
    triton-inflight_batcher_llm-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_inflight_batcher_llm
  )
else()
  set_target_properties(
    triton-inflight_batcher_llm-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_inflight_batcher_llm
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_inflight_batcher_llm.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_inflight_batcher_llm.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/InflightBatcherLLMBackend)

install(
  TARGETS
    triton-inflight_batcher_llm-backend
  EXPORT
    triton-inflight_batcher_llm-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/inflight_batcher_llm
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/inflight_batcher_llm
)

install(
  EXPORT
    triton-inflight_batcher_llm-backend-targets
  FILE
    InflightBatcherLLMBackendTargets.cmake
  NAMESPACE
    InflightBatcherLLMBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/InflightBatcherLLMBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/InflightBatcherLLMBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/InflightBatcherLLMBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-inflight_batcher_llm-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/InflightBatcherLLMBackendTargets.cmake
  NAMESPACE InflightBatcherLLMBackend::
)

export(PACKAGE InflightBatcherLLMBackend)
