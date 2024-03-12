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

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <unordered_set>
#include <utils.h>

namespace triton::backend::inflight_batcher_llm
{

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
    using InferenceRequest = tensorrt_llm::batch_manager::InferenceRequest;
    using NamedTensor = tensorrt_llm::batch_manager::NamedTensor;

public:
    WorkItem(TRITONBACKEND_Request* request, bool isDecoupled);
    WorkItem(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled);
    WorkItem(std::shared_ptr<InferenceRequest> ir, uint64_t RequestId);
    ~WorkItem();

    TRITONBACKEND_ResponseFactory* response_factory();

    uint64_t requestId() const;

    std::shared_ptr<InferenceRequest> getInferenceRequest() const;

    bool hasOutputName(std::string const& outputName);

    /// timestamp storage for Triton base metrics
    struct Timestamps
    {
        uint64_t exec_start_ns = 0;
        uint64_t compute_start_ns = 0;
        uint64_t compute_end_ns = 0;
        uint64_t exec_end_ns = 0;

        void Reset()
        {
            exec_start_ns = 0;
            compute_start_ns = 0;
            compute_end_ns = 0;
            exec_end_ns = 0;
        }
    };

    Timestamps& getTimestamps();

    TRITONBACKEND_Request* getTritonInferenceRequest() const;

    TRITONSERVER_Error* reportBaseMetrics(TRITONBACKEND_ModelInstance* model_instance, TRITONSERVER_Error* err);

private:
    // Convert Trition request to trtllm InferenceRequest
    static std::shared_ptr<InferenceRequest> createInferenceRequest(
        TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled);

    void Initialize(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled);

    std::shared_ptr<InferenceRequest> mInferenceRequest;
    TRITONBACKEND_ResponseFactory* factory_ptr_;
    uint64_t mRequestId;
    std::unordered_set<std::string> mRequestOutputNames;

    Timestamps mTimestamps;
    TRITONBACKEND_Request* mTritonInferenceRequest;
};

} // namespace triton::backend::inflight_batcher_llm
