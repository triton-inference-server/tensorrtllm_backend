// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils.h"
#include <cassert>

namespace triton::backend::inflight_batcher_llm::utils
{

nvinfer1::DataType to_trt_datatype(TRITONSERVER_DataType data_type)
{
    if (data_type == TRITONSERVER_TYPE_INVALID)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_BOOL)
    {
        return nvinfer1::DataType::kBOOL;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT8)
    {
        return nvinfer1::DataType::kUINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT16)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_UINT32)
    {
        return nvinfer1::DataType::kINT32;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT64)
    {
        return nvinfer1::DataType::kINT64;
    }
    else if (data_type == TRITONSERVER_TYPE_INT8)
    {
        return nvinfer1::DataType::kINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_INT16)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_INT32)
    {
        return nvinfer1::DataType::kINT32;
    }
    else if (data_type == TRITONSERVER_TYPE_INT64)
    {
        return nvinfer1::DataType::kINT64;
    }
    else if (data_type == TRITONSERVER_TYPE_FP16)
    {
        return nvinfer1::DataType::kHALF;
    }
    else if (data_type == TRITONSERVER_TYPE_FP32)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    else if (data_type == TRITONSERVER_TYPE_FP64)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_BYTES)
    {
        return nvinfer1::DataType::kINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_BF16)
    {
        return nvinfer1::DataType::kBF16;
    }
    else
    {
        assert(false);
    }
    return nvinfer1::DataType(0);
}

TRITONSERVER_DataType to_triton_datatype(nvinfer1::DataType data_type)
{
    if (data_type == nvinfer1::DataType::kBOOL)
    {
        return TRITONSERVER_TYPE_BOOL;
    }
    else if (data_type == nvinfer1::DataType::kUINT8)
    {
        return TRITONSERVER_TYPE_UINT8;
    }
    else if (data_type == nvinfer1::DataType::kHALF)
    {
        return TRITONSERVER_TYPE_BF16;
    }
    else if (data_type == nvinfer1::DataType::kINT8)
    {
        return TRITONSERVER_TYPE_INT8;
    }
    else if (data_type == nvinfer1::DataType::kINT32)
    {
        return TRITONSERVER_TYPE_INT32;
    }
    else if (data_type == nvinfer1::DataType::kINT64)
    {
        return TRITONSERVER_TYPE_INT64;
    }
    else if (data_type == nvinfer1::DataType::kFLOAT)
    {
        return TRITONSERVER_TYPE_FP32;
    }
    else if (data_type == nvinfer1::DataType::kBF16)
    {
        return TRITONSERVER_TYPE_BF16;
    }
    else
    {
        return TRITONSERVER_TYPE_INVALID;
    }
}

uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap)
{
    const char* charRequestId;
    TRITONBACKEND_RequestId(request, &charRequestId);
    uint64_t requestId = 0;
    if (charRequestId != nullptr)
    {
        std::string strRequestId(charRequestId);
        if (!strRequestId.empty())
        {
            try
            {
                requestId = stoul(strRequestId);
            }
            catch (const std::exception& e)
            {
                std::hash<std::string> hasher;
                requestId = hasher(strRequestId);

                // Check for hash collisions
                // If requestID already exists in the map with the same string, increment the ID and check again
                for (auto it = requestIdStrMap.find(requestId);
                     it != requestIdStrMap.end() && it->second != strRequestId;)
                {
                    requestId++;
                }
            }
            requestIdStrMap.insert({requestId, strRequestId});
        }
    }

    return requestId;
}

std::string getRequestIdStr(uint64_t requestId, std::unordered_map<uint64_t, std::string> const& requestIdStrMap)
{
    auto it = requestIdStrMap.find(requestId);
    if (it != requestIdStrMap.end())
    {
        return it->second;
    }
    return std::to_string(requestId);
}

std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request)
{
    std::unordered_set<std::string> outputNames;
    uint32_t outputCount;
    LOG_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &outputCount), "Error getting request output count");
    for (size_t i = 0; i < outputCount; ++i)
    {
        const char* name;
        LOG_IF_ERROR(TRITONBACKEND_RequestOutputName(request, i, &name), "Error getting request output name");
        std::string name_s(name);
        outputNames.insert(std::move(name_s));
    }
    return outputNames;
}

bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, const std::string& inputTensorName)
{
    // Get stop signal from the request
    TRITONBACKEND_Input* input;
    TRITONSERVER_Error* error = TRITONBACKEND_RequestInput(request, inputTensorName.c_str(), &input);
    if (error)
    {
        // If the user does not provide input "stop", then regard the request as
        // unstopped
        std::string msg
            = "ModelInstanceState::getRequestBooleanInputTensor: user "
              "did not not provide "
            + inputTensorName + " input for the request";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg.c_str());
        TRITONSERVER_ErrorDelete(error);
        return false;
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
        ("ModelInstanceState::getRequestStopSignal: buffer_count = " + std::to_string(buffer_count)).c_str());

    const void* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));

    bool boolean = *reinterpret_cast<const bool*>(buffer);

    return boolean;
}

} // namespace triton::backend::inflight_batcher_llm::utils
