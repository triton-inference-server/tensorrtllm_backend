# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set_ifndef(TRTLLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_llm)

include(${TRTLLM_DIR}/cpp/cmake/modules/find_library_create_target.cmake)

project(tritontensorrtllmbackend LANGUAGES C CXX)

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
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
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(BUILD_TESTS "Build Google tests" OFF)

if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(
    FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
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

#
# The backend must be built into a shared library. Use an ldscript to hide all
# symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_tensorrtllm.ldscript
               libtriton_tensorrtllm.ldscript COPYONLY)

set(COMMON_SRCS
    src/work_item.cc src/work_items_queue.cc src/model_instance_state.cc
    src/model_state.cc src/utils.cc src/inference_answer.cc)

add_library(triton-tensorrt-llm-common SHARED ${COMMON_SRCS})

set(BACKEND_SRCS src/libtensorrtllm.cc src/orchestrator.cc)

add_library(triton-tensorrt-llm-backend SHARED ${BACKEND_SRCS})

set(WORKER_SRCS src/worker.cc)

add_executable(triton-tensorrt-llm-worker ${WORKER_SRCS})

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
  triton-tensorrt-llm-common
  PUBLIC ${TRTLLM_DIR}/cpp
         ${TRTLLM_DIR}/cpp/include
         ${CMAKE_CURRENT_SOURCE_DIR}/src
         ${CUDA_INCLUDE_DIRS}
         ${CUDNN_ROOT_DIR}/include
         ${NCCL_INCLUDE_DIR}
         ${3RDPARTY_DIR}/cutlass/include
         ${MPI_INCLUDE_PATH}
         ${COMMON_HEADER_DIR})

target_compile_features(triton-tensorrt-llm-common PRIVATE cxx_std_17)
target_compile_features(triton-tensorrt-llm-backend PRIVATE cxx_std_17)
target_compile_features(triton-tensorrt-llm-worker PRIVATE cxx_std_17)

set(COMPILE_OPTIONS
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-declarations
    -Wno-type-limits>
    $<$<CXX_COMPILER_ID:MSVC>:/Wall
    /D_WIN32_WINNT=0x0A00
    /EHsc>)

target_compile_options(triton-tensorrt-llm-common PRIVATE ${COMPILE_OPTIONS})
target_compile_options(triton-tensorrt-llm-backend PRIVATE ${COMPILE_OPTIONS})
target_compile_options(triton-tensorrt-llm-worker PRIVATE ${COMPILE_OPTIONS})

add_library(tensorrt_llm SHARED IMPORTED)
set_property(
  TARGET tensorrt_llm
  PROPERTY IMPORTED_LOCATION
           "${TRTLLM_DIR}/cpp/build/tensorrt_llm/libtensorrt_llm.so")

add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
  TARGET nvinfer_plugin_tensorrt_llm
  PROPERTY
    IMPORTED_LOCATION
    "${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)

if(TRITON_ENABLE_METRICS)
  list(APPEND REPORTER_SRCS
       src/custom_metrics_reporter/custom_metrics_reporter.cc)
  list(APPEND REPORTER_HDRS
       src/custom_metrics_reporter/custom_metrics_reporter.h)

  add_library(triton-custom-metrics-reporter-library EXCLUDE_FROM_ALL
              ${REPORTER_SRCS} ${REPORTER_HDRS})
  target_compile_features(triton-custom-metrics-reporter-library
                          PRIVATE cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(triton-custom-metrics-reporter-library
                           PRIVATE /W1 /D_WIN32_WINNT=0x0A00 /EHsc)
  else()
    target_compile_options(
      triton-custom-metrics-reporter-library
      PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations
              -Werror)
  endif()

  set_target_properties(triton-custom-metrics-reporter-library
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(
    triton-custom-metrics-reporter-library
    PUBLIC triton-common-json # from repo-common
           triton-common-logging # from repo-common
           triton-core-serverapi # from repo-core
           triton-core-serverstub # from repo-core
           triton-backend-utils # from repo-backend
           tensorrt_llm)

  target_compile_definitions(triton-tensorrt-llm-common
                             PRIVATE TRITON_ENABLE_METRICS=1)
  target_link_libraries(triton-tensorrt-llm-common
                        PRIVATE triton-custom-metrics-reporter-library)
endif()

if(TRITON_BUILD)

  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    execute_process(
      WORKING_DIRECTORY ${TRTLLM_DIR}
      COMMAND bash -x docker/common/install_pytorch.sh pypi COMMAND_ECHO STDOUT
              COMMAND_ERROR_IS_FATAL ANY)
  else()
    execute_process(
      WORKING_DIRECTORY ${TRTLLM_DIR}
      COMMAND bash -x docker/common/install_pytorch.sh src_non_cxx11_abi
              COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY)
  endif() # CMAKE_HOST_SYSTEM_PROCESSOR

  execute_process(
    WORKING_DIRECTORY ${TRTLLM_DIR}
    COMMAND python3 scripts/build_wheel.py --trt_root /usr/local/tensorrt
            COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY)

endif() # TRITON_BUILD

target_link_libraries(
  triton-tensorrt-llm-common
  PUBLIC tensorrt_llm
         triton-core-serverapi # from repo-core
         triton-core-backendapi # from repo-core
         triton-core-serverstub # from repo-core
         triton-backend-utils # from repo-backend
         ${MPI_LIBRARIES}
         ${CUDA_LIBRARIES}
         nvinfer
         nvinfer_plugin_tensorrt_llm)

target_link_libraries(triton-tensorrt-llm-backend
                      PRIVATE triton-tensorrt-llm-common)
target_link_libraries(triton-tensorrt-llm-worker
                      PRIVATE triton-tensorrt-llm-common)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2)

FetchContent_MakeAvailable(json)

target_link_libraries(triton-tensorrt-llm-common
                      PRIVATE nlohmann_json::nlohmann_json)

if(WIN32)
  set_target_properties(
    triton-tensorrt-llm-backend PROPERTIES POSITION_INDEPENDENT_CODE ON
                                           OUTPUT_NAME triton_tensorrtllm)
  set_target_properties(
    triton-tensorrt-llm-worker PROPERTIES POSITION_INDEPENDENT_CODE ON
                                          OUTPUT_NAME triton_tensorrtllm_worker)
  set_target_properties(
    triton-tensorrt-llm-common PROPERTIES POSITION_INDEPENDENT_CODE ON
                                          OUTPUT_NAME triton_tensorrtllm_common)
else()
  set_target_properties(
    triton-tensorrt-llm-backend
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME triton_tensorrtllm
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorrtllm.ldscript
      LINK_FLAGS
      "-Wl,--version-script libtriton_tensorrtllm.ldscript -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined"
  )
  set_target_properties(triton-tensorrt-llm-worker
                        PROPERTIES OUTPUT_NAME triton_tensorrtllm_worker)
  set_target_properties(
    triton-tensorrt-llm-common PROPERTIES POSITION_INDEPENDENT_CODE ON
                                          OUTPUT_NAME triton_tensorrtllm_common)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
