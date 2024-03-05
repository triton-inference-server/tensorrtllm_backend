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

#include "work_item.h"
#include <map>

namespace triton::backend::inflight_batcher_llm
{

WorkItem::WorkItem(TRITONBACKEND_Request* request, bool isDecoupled)
{
    uint64_t requestId = (rand() % INT64_MAX) + 1;
    Initialize(request, requestId, isDecoupled);
}

WorkItem::WorkItem(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
{
    Initialize(request, requestId, isDecoupled);
}

WorkItem::WorkItem(std::shared_ptr<tensorrt_llm::batch_manager::InferenceRequest> ir, uint64_t RequestId)
    : mInferenceRequest(ir)
    , mRequestId(RequestId)
{
    factory_ptr_ = nullptr;
}

WorkItem::~WorkItem()
{
    if (factory_ptr_ != nullptr)
    {
        TRITONBACKEND_ResponseFactoryDelete(factory_ptr_);
    }
}

TRITONBACKEND_ResponseFactory* WorkItem::response_factory()
{
    assert(factory_ptr_ != nullptr);
    return factory_ptr_;
}

uint64_t WorkItem::requestId() const
{
    return mRequestId;
}

std::shared_ptr<tensorrt_llm::batch_manager::InferenceRequest> WorkItem::getInferenceRequest() const
{
    return mInferenceRequest;
}

bool WorkItem::hasOutputName(const std::string& outputName)
{
    return (mRequestOutputNames.find(outputName) != mRequestOutputNames.end());
}

std::shared_ptr<tensorrt_llm::batch_manager::InferenceRequest> WorkItem::createInferenceRequest(
    TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
{
    auto inferenceRequest = std::make_shared<InferenceRequest>(requestId);

    // Extract input tensors
    std::map<std::string, NamedTensor> input_tensors;
    uint32_t num_inputs;
    LOG_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &num_inputs), "Error getting input count");
    for (uint32_t idx = 0; idx < num_inputs; ++idx)
    {
        TRITONBACKEND_Input* input = 0L;
        TRITONBACKEND_RequestInputByIndex(request, idx, &input);

        const char* input_name = 0L;
        TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_INVALID;
        const int64_t* shape = 0L;
        uint32_t dims_count = 0;
        uint64_t byte_size = 0;
        uint32_t buffer_count = 0;
        TRITONBACKEND_InputProperties(input, &input_name, &data_type, &shape, &dims_count, &byte_size, &buffer_count);

        if (std::string(input_name) == "START" || std::string(input_name) == "CORRID"
            || std::string(input_name) == "END" || std::string(input_name) == kStopInputTensorName
            || std::string(input_name) == kStreamingInputTensorName)
        {
            continue;
        }

        std::vector<int64_t> shapev;
        for (uint32_t i = 0; i < dims_count; ++i)
        {
            shapev.push_back(shape[i]);
        }

        NamedTensor t(utils::to_trt_datatype(data_type), shapev, input_name);
        uint64_t buffer_offset = 0;
        for (int64_t buffer_id = 0; buffer_id < buffer_count; ++buffer_id)
        {
            const void* buffer = 0L;
            uint64_t buffer_byte_size = 0;
            TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            TRITONBACKEND_InputBuffer(input, buffer_id, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);
            assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));
            // TODO: Do we need to handle GPU mem input buffers??
            std::memcpy(static_cast<char*>(t.tensor->data()) + buffer_offset, buffer, buffer_byte_size);
            buffer_offset += buffer_byte_size;
        }

        inferenceRequest->emplaceInputTensor(t.name, std::move(t.tensor));
    }

    bool streamingFlag = utils::getRequestBooleanInputTensor(request, kStreamingInputTensorName);
    inferenceRequest->setIsStreaming(streamingFlag);

    if (streamingFlag && !isDecoupled)
    {
        throw std::runtime_error(
            "Streaming is only supported if model is "
            "deployed using decoupled mode.");
    }

    return inferenceRequest;
}

void WorkItem::Initialize(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
{
    mRequestId = requestId;
    mInferenceRequest = createInferenceRequest(request, requestId, isDecoupled);
    mRequestOutputNames = utils::getRequestOutputNames(request);

    // Create response factory for this request
    TRITONBACKEND_ResponseFactoryNew(&factory_ptr_, request);

    // Store an unconverted version of the TRITONBACKEND_Request to be used
    // to release the request when the base metrics have been reported
    mTritonInferenceRequest = request;
    mTimestamps.Reset();
}

WorkItem::Timestamps& WorkItem::getTimestamps()
{
    return mTimestamps;
}

TRITONBACKEND_Request* WorkItem::getTritonInferenceRequest() const
{
    return mTritonInferenceRequest;
}

TRITONSERVER_Error* WorkItem::reportBaseMetrics(TRITONBACKEND_ModelInstance* model_instance, TRITONSERVER_Error* err)
{
    SET_TIMESTAMP(mTimestamps.exec_end_ns);
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(model_instance, mTritonInferenceRequest,
        (err == nullptr), mTimestamps.exec_start_ns, mTimestamps.compute_start_ns, mTimestamps.compute_end_ns,
        mTimestamps.exec_end_ns));

    // For now we will assume a batch size of 1 for each request. This may change in the future but for
    // now it seems that even when requests are dynamically batched together each workItem is associated
    // with its own request object and is handled independently due to the nature of IFB.
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(model_instance, 1 /* batch size */,
        mTimestamps.exec_start_ns, mTimestamps.compute_start_ns, mTimestamps.compute_end_ns, mTimestamps.exec_end_ns));
    return nullptr; // success
}

} // namespace triton::backend::inflight_batcher_llm
