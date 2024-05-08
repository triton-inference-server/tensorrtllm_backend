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

#include <map>
#include <queue>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/runtime/decodingMode.h"

#include "model_state.h"

#ifdef TRITON_ENABLE_METRICS
#include "custom_metrics_reporter/custom_metrics_reporter.h"
#endif

using namespace tensorrt_llm;
using namespace tensorrt_llm::batch_manager;

namespace triton::backend::inflight_batcher_llm
{

/// @brief Struct to hold configs that is will be used later when creating the executor requests
struct InstanceSpecificConfig
{
    bool excludeInputFromOutput;
    int cancellationCheckPeriodMs;
};

/// @brief Per-request data stored for handling requests
struct RequestData
{
    TRITONBACKEND_ResponseFactory* factory;
    TRITONBACKEND_Request* tritonRequest;
    std::string tritonRequestId;
    int64_t inputTokensSize;
    executor::SizeType32 beamWidth;
};

//
// ModelInstanceState
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
//
class ModelInstanceState
{
    using DecodingMode = tensorrt_llm::runtime::DecodingMode;
    using InferenceRequest = tensorrt_llm::batch_manager::InferenceRequest;
    using NamedTensor = tensorrt_llm::batch_manager::NamedTensor;
    using TrtGptModelType = tensorrt_llm::batch_manager::TrtGptModelType;

public:
    // number of cpu workers used to move weights host cache to gpu cache
    static constexpr SizeType32 kPeftCacheNumEnsureWorkers = 4;
    // number of cuda streams used for H2D copies of peft cache pages
    static constexpr SizeType32 kPeftCacheNumCopyStreams = 4;
    // number of cpu workers used to load weight into host cache
    static constexpr SizeType32 kPeftCacheNumPutWorkers = 4;

    /// @brief Create a ModelInstanceObject
    static TRITONSERVER_Error* Create(
        ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state);

    virtual ~ModelInstanceState()
    {
        mStopWaitForResponse = true;
        mWaitForResponseThread.join();

        mStopWaitForStats = true;
        mWaitForStatsThread.join();

        mStopWaitForCancel = true;
        mWaitForCancelThread.join();
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

    /// @brief Add the request to the executor
    void enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count);

private:
    /// @brief Get batching type
    executor::BatchingType getBatchingTypeFromParams();

    /// @brief Get kv cache config
    executor::KvCacheConfig getKvCacheConfigFromParams();

    /// @brief Get scheduler config
    executor::SchedulerConfig getSchedulerConfigFromParams(bool enableChunkedContext);

    /// @brief Get peft config
    executor::PeftCacheConfig getPeftCacheConfigFromParams();

    /// @brief Get parallel config
    executor::ParallelConfig getParallelConfigFromParams();

    /// @brief Get executor config
    executor::ExecutorConfig getExecutorConfigFromParams();

    /// @brief Constructor
    ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance);

    ModelState* model_state_;
    TRITONBACKEND_ModelInstance* modelInstance_;

    std::string mModelPath;

    /// @brief Send a response during enqueue
    void sendEnqueueResponse(TRITONBACKEND_Request* request, TRITONSERVER_Error* error);

    /// @brief Cancel a request
    bool handleStopRequest(TRITONBACKEND_Request* request, std::string const& tritonRequestId);

    /// @brief Create an executor::Request from input tensors
    static executor::Request createExecutorRequest(
        TRITONBACKEND_Request* request, bool excludeInputFromOutput, bool isDecoupled);

    /// @brief Fill in a triton response based on executor response
    std::tuple<TRITONBACKEND_Response*, bool, TRITONSERVER_Error*> fillTritonResponse(
        TRITONBACKEND_ResponseFactory* factory, executor::Response const& response, RequestData const& requestData);

    /// @brief TRT-LLM Executor that handles requests
    std::unique_ptr<executor::Executor> mExecutor;
    /// @brief Config to be used when sending requests to executor
    InstanceSpecificConfig mInstanceSpecificConfig;

    /// @brief Retrieve responses from the executor
    void WaitForResponse();
    /// @brief The thread for WaitForResponse() to run
    std::thread mWaitForResponseThread;
    /// @brief Flag to stop the WaitForResponse thread when the model instance is being destroyed
    bool mStopWaitForResponse;

    /// @brief Retrieve stats from the executor
    void WaitForStats();
    /// @brief The thread for WaitForStats() to run
    std::thread mWaitForStatsThread;
    /// @brief Flag to stop the WaitForStats thread when the model instance is being destroyed
    bool mStopWaitForStats;

    /// @brief Cancel a request for executor if it is marked as cancelled by Triton backend
    void WaitForCancel();
    /// @brief The thread for WaitForCancel() to run
    std::thread mWaitForCancelThread;
    /// @brief Flag to stop the WaitForCancel thread when the model instance is being destroyed
    bool mStopWaitForCancel;

    std::unordered_map<executor::IdType, RequestData> mRequestIdToRequestData;
    std::unordered_map<std::string, executor::IdType> mTritonRequestIdToRequestId;
    std::mutex mRequestIdToRequestDataMutex;

#ifdef TRITON_ENABLE_METRICS
    std::unique_ptr<custom_metrics_reporter::CustomMetricsReporter> custom_metrics_reporter_;
#endif
};

} // namespace triton::backend::inflight_batcher_llm
