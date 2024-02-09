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

#include <nlohmann/json.hpp>
#include <unordered_map>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/batch_manager/schedulerPolicy.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"

#include "model_state.h"
#include "work_item.h"
#include "work_items_queue.h"

#ifdef TRITON_ENABLE_METRICS
#include "custom_metrics_reporter/custom_metrics_reporter.h"
#endif

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::batch_scheduler;

namespace triton::backend::inflight_batcher_llm
{

//
// ModelInstanceState
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
//

class ModelInstanceState
{
    using InferenceRequest = tensorrt_llm::batch_manager::InferenceRequest;
    using NamedTensor = tensorrt_llm::batch_manager::NamedTensor;
    using TrtGptModelType = tensorrt_llm::batch_manager::TrtGptModelType;

public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state);

    virtual ~ModelInstanceState()
    {
        // terminate decoupled execution loop
        {
            mWorkItemsQueue->clear();
        }

        // signal batch manager to stop processing the work items queue
        {
            mBatchManager->shutdown();
        }
    }

    // Get the state of the model that corresponds to this instance.
    ModelState* StateForModel() const
    {
        return model_state_;
    }

    bool isDecoupled() const
    {
        return model_state_->IsDecoupled();
    }

    /// @brief Add the request to the WorkItemsQueue
    void enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count);

    /// @brief  Callback passed to GptManager to get new inference requests
    /// @return Up to max_num_requests inference requests.
    std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(const int max_num_requests);

    /// @brief  Callback passed to GptManager to send responses back to client
    void sendResponse(uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response,
        const std::string& errMsg);
    /// @brief Callback passed to GptManager to get ids of stopped requests
    std::unordered_set<uint64_t> pollStopSignals();

    /// @brief  Callback passed to GptManager to print stats
    void logStats(const std::string& s);

    /// @brief Method that sends Triton response back to client
    TRITONSERVER_Error* sendTritonResponse(std::shared_ptr<WorkItem> workItem,
        std::list<NamedTensor> const& response_tensors, bool final_response, const std::string& errMsg);

private:
    /// @brief Constructor
    ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance);

    ModelState* model_state_;
    TRITONBACKEND_ModelInstance* modelInstance_;

    TrtGptModelType mTrtGptModelType;
    std::string mModelPath;

    std::shared_ptr<GptManager> mBatchManager;
    std::unique_ptr<WorkItemsQueue> mWorkItemsQueue;

    std::unordered_map<uint64_t, std::string> mRequestIdStrMap;
#ifdef TRITON_ENABLE_METRICS
    std::unique_ptr<custom_metrics_reporter::CustomMetricsReporter> custom_metrics_reporter_;
#endif

    bool mHasActiveRequests;
};

} // namespace triton::backend::inflight_batcher_llm
