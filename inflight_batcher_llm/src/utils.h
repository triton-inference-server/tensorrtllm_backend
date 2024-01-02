// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#define _GLIBCXX_USE_CXX11_ABI 0

#include "NvInfer.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <string>
#include <unordered_set>

namespace triton::backend::inflight_batcher_llm
{
inline static const std::string kStopInputTensorName = "stop";
inline static const std::string kStreamingInputTensorName = "streaming";

namespace utils
{

/// @brief  Convert Triton datatype to TRT datatype
nvinfer1::DataType to_trt_datatype(TRITONSERVER_DataType data_type);

/// @brief  Convert TRT datatype to Triton datatype
TRITONSERVER_DataType to_triton_datatype(nvinfer1::DataType data_type);

/// @brief get the requestId of the request and update requestIdStrMap
/// @return Returns 0 if not specified. Throws an error if request_id cannot be convert to uint64_t
uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap);

/// @brief get the original requestId string from the uint64_t requestId
/// @return If uint64_t id is not present in requestIdStrMap, returns std::to_string(requestId)
std::string getRequestIdStr(uint64_t requestId, std::unordered_map<uint64_t, std::string> const& requestIdStrMap);

/// @brief Get the requested output names
std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request);

/// @brief Get the value of a boolean tensor
bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, const std::string& inputTensorName);

} // namespace utils
} // namespace triton::backend::inflight_batcher_llm
