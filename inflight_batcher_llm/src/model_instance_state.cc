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
#define _GLIBCXX_USE_CXX11_ABI 0

#include "model_instance_state.h"

namespace triton::backend::inflight_batcher_llm
{

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state)
{
    try
    {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    }
    catch (const std::exception& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelInstanceState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : model_state_(model_state)
    , mIsDecoupled(false)
{
    // Note: std::string::compare fails this test (always return non-zero
    // value). Using old school strcmp instead.
    auto gpt_model_type = model_state_->GetParameter<std::string>("gpt_model_type");
    if (gpt_model_type == "V1" || gpt_model_type == "v1")
    {
        mTrtGptModelType = TrtGptModelType::V1;
    }
    else if (gpt_model_type == "inflight_batching")
    {
        mTrtGptModelType = TrtGptModelType::InflightBatching;
    }
    else if (gpt_model_type == "inflight_fused_batching")
    {
        mTrtGptModelType = TrtGptModelType::InflightFusedBatching;
    }
    else
    {
        throw std::runtime_error(
            "Invalid gpt_model_type. Must be "
            "v1/inflight_batching/inflight_fused_batching.");
    }

    // Check if model is in decoupled mode:
    triton::common::TritonJson::Value transaction_policy;
    model_state_->GetModelConfig().MemberAsObject("model_transaction_policy", &transaction_policy);
    transaction_policy.MemberAsBool("decoupled", &mIsDecoupled);

    // Note: std::string::compare fails this test (always return non-zero
    // value). Using old school strcmp instead.
    mModelPath = model_state_->GetParameter<std::string>("gpt_model_path");
    auto configPath = mModelPath + "/config.json";
    std::ifstream jsonStream(configPath);

    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);

    int32_t maxBeamWidth = 1;
    try
    {
        maxBeamWidth = model_state_->GetParameter<int32_t>("max_beam_width");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("max_beam_width is not specified, will use default value of 1");
    }

    std::optional<int32_t> maxTokensInPagedKvCache = std::nullopt;
    try
    {
        maxTokensInPagedKvCache = model_state_->GetParameter<int32_t>("max_tokens_in_paged_kv_cache");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "max_tokens_in_paged_kv_cache is not specified, will "
            "use default value");
    }

    auto schedulerPolicy = SchedulerPolicy::GUARANTEED_NO_EVICT;
    try
    {
        std::string schedulerPolicyStr = model_state_->GetParameter<std::string>("batch_scheduler_policy");
        if (schedulerPolicyStr == "max_utilization")
        {
            schedulerPolicy = SchedulerPolicy::MAX_UTILIZATION;
        }
        else if (schedulerPolicyStr == "guaranteed_no_evict")
        {
            schedulerPolicy = SchedulerPolicy::GUARANTEED_NO_EVICT;
        }
        else
        {
            throw std::runtime_error(
                "batch_scheduler_policy parameter was not found or is invalid "
                "(must be max_utilization or guaranteed_no_evict)");
        }
    }
    catch (const std::exception& e)
    {
        TLLM_LOG_WARNING(e.what());
    }

    if (mIsDecoupled && schedulerPolicy != SchedulerPolicy::GUARANTEED_NO_EVICT)
    {
        TLLM_LOG_WARNING(
            "The batch scheduler policy will be set to guaranteed_no_evict"
            "since the backend operates in decoupled mode");
        schedulerPolicy = SchedulerPolicy::GUARANTEED_NO_EVICT;
    }

    std::optional<float> kvCacheFreeGpuMemFraction = std::nullopt;
    try
    {
        kvCacheFreeGpuMemFraction = model_state_->GetParameter<float>("kv_cache_free_gpu_mem_fraction");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "kv_cache_free_gpu_mem_fraction is not specified, will use default value of 0.85 or "
            "max_tokens_in_paged_kv_cache");
    }

    std::optional<int32_t> maxNumSequences = std::nullopt;
    try
    {
        maxNumSequences = model_state_->GetParameter<int32_t>("max_num_sequences");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("max_num_sequences is not specified, will be set to the TRT engine max_batch_size");
    }

    bool enableTrtOverlap = true;
    try
    {
        enableTrtOverlap = model_state_->GetParameter<bool>("enable_trt_overlap");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_trt_overlap is not specified, will be set to true");
    }

    bool excludeInputInOutput = false;
    try
    {
        excludeInputInOutput = model_state_->GetParameter<bool>("exclude_input_in_output");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("exclude_input_in_output is not specified, will be set to false");
    }

    std::optional<int32_t> maxKvCacheLength = std::nullopt;
    try
    {
        maxKvCacheLength = model_state_->GetParameter<int32_t>("max_kv_cache_length");
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "max_kv_cache_length is not specified, will "
            "use default value");
    }

    TrtGptModelOptionalParams optionalParams;
    optionalParams.maxNumSequences = maxNumSequences;
    optionalParams.kvCacheConfig.maxTokens = maxTokensInPagedKvCache;
    optionalParams.kvCacheConfig.freeGpuMemoryFraction = kvCacheFreeGpuMemFraction;
    optionalParams.kvCacheConfig.maxKvCacheLength = maxKvCacheLength;
    optionalParams.enableTrtOverlap = enableTrtOverlap;

    mBatchManager = std::make_shared<GptManager>(
        mModelPath, mTrtGptModelType, maxBeamWidth, schedulerPolicy,
        [this](int max_num_requests) { return get_inference_requests(max_num_requests); },
        [this](uint64_t requestId, std::list<NamedTensor> response_tensors, bool final_response,
            const std::string& errMsg) { return sendResponse(requestId, response_tensors, final_response, errMsg); },
        [this]() { return pollStopSignals(); }, [this](const std::string& s) { return logStats(s); }, optionalParams,
        std::nullopt, std::nullopt, excludeInputInOutput);

    if (getCommWorldRank() != 0)
    {
        while (true)
        {
        }
    }
}

// For stop requests, or in case of error during enqueue, we need to send a
// response to the client
void ModelInstanceState::sendEnqueueResponse(TRITONBACKEND_Request* request, const std::string& errMsg)
{
    TRITONBACKEND_ResponseFactory* factory_ptr;
    // Create response factory for this request
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request), "Cannot create response factory");

    TRITONSERVER_Error* err = nullptr;
    if (!errMsg.empty())
    {
        TLLM_LOG_ERROR(errMsg);
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errMsg.c_str());
    }
    TRITONBACKEND_Response* response;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&response, factory_ptr), "Cannot create response");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err), "Cannot send response");
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory_ptr), "Cannot delete response factory");
}

void ModelInstanceState::enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count, bool isDecoupled)
{
    std::vector<std::pair<uint64_t, TRITONBACKEND_Request*>> requestsToPush;

    for (uint32_t r = 0; r < request_count; ++r)
    {
        TRITONBACKEND_Request* request = requests[r];
        try
        {
            auto requestId = utils::getRequestId(request);
            bool stopRequest = utils::getRequestBooleanInputTensor(request, kStopInputTensorName);

            if (stopRequest)
            {
                if (requestId != 0)
                {
                    // Check if request is in progress or in queue, if not ignore
                    mWorkItemsQueue.stopWorkItem(requestId);
                    // Send a response back to client for stop request
                    sendEnqueueResponse(request);
                }
                else
                {
                    throw std::runtime_error("Cannot send stop request without specifying a request_id");
                }
            }
            else
            {
                requestsToPush.emplace_back(requestId, request);
            }
        }
        catch (const std::exception& e)
        {
            // In case of error, no work item is added to queue, so response
            // callback needs to be called
            sendEnqueueResponse(request, e.what());
        }
    }

    auto exceptions = mWorkItemsQueue.pushBatch(requestsToPush, isDecoupled);

    for (uint32_t r = 0; r < requestsToPush.size(); ++r)
    {
        auto request = requestsToPush.at(r).second;
        auto e = exceptions.at(r);
        if (e)
        {
            sendEnqueueResponse(request, e->what());
        }
    }

    return;
}

// Return up to max_num_requests inference requests.
std::list<std::shared_ptr<InferenceRequest>> ModelInstanceState::get_inference_requests(const int max_num_requests)
{
    std::list<std::shared_ptr<InferenceRequest>> rval;
    if (max_num_requests <= 0)
    {
        return rval;
    }

    auto world_size = getCommWorldSize();
    auto rank = getCommWorldRank();
    if (rank == 0)
    {
        auto numPendingWorkItems = mWorkItemsQueue.numPendingWorkItems();
        // Loop over the pending work items and include at most `max_num_requests`
        for (size_t i = 0; i < numPendingWorkItems && static_cast<int>(rval.size()) < max_num_requests; ++i)
        {
            auto [workItem, stoppedRequest] = mWorkItemsQueue.pop();

            if (workItem)
            {
                if (!stoppedRequest)
                {
                    rval.emplace_back(workItem->getInferenceRequest());
                }
                else
                {
                    std::string warnStr = std::string("request Id ") + std::to_string(workItem->requestId())
                        + std::string(" has been stopped. Request is ignored.");
                    TLLM_LOG_WARNING(warnStr);
                    sendTritonResponse(workItem, {}, true, warnStr);
                }
            }
        }

        if (world_size > 1)
        {
            int64_t num_new_work_items = rval.size();
            bcast(&num_new_work_items, 1, MPI_TYPE_INT64_T, 0, COMM_WORLD);

            if (num_new_work_items > 0)
            {
                std::vector<int64_t> packed;
                for (auto ir : rval)
                {
                    auto vpacked = ir->serialize();
                    packed.push_back(static_cast<int64_t>(vpacked.size()));
                    packed.insert(packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                }
                bcast(packed, 0, COMM_WORLD);
            }
        }
    }
    else
    {
        // subordinate ranks hang until master rank sends work
        int64_t num_new_work_items;
        bcast(&num_new_work_items, 1, MPI_TYPE_INT64_T, 0, COMM_WORLD);
        if (num_new_work_items > 0)
        {
            std::vector<int64_t> packed;
            bcast(packed, 0, COMM_WORLD);
            int64_t* packed_ptr = packed.data();
            for (int64_t count = 0; count < num_new_work_items; ++count)
            {
                int64_t n = *(packed_ptr++);
                auto ir = InferenceRequest::deserialize(packed_ptr);
                packed_ptr += n;
                rval.emplace_back(ir);
            }
        }
    }
    return rval;
}

void ModelInstanceState::sendResponse(
    uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response, const std::string& errMsg)
{
    if (getCommWorldRank() == 0)
    {
        std::string errStr = std::string("Failed to send Triton response for requestId: ") + std::to_string(requestId);
        try
        {
            auto workItem = mWorkItemsQueue.getInProgressWorkItem(requestId);
            auto tritonErr = sendTritonResponse(workItem, response_tensors, final_response, errMsg);
            LOG_IF_ERROR(tritonErr, errStr);
        }
        catch (const std::exception& e)
        {
            TLLM_LOG_ERROR(errStr);
        }
    }
}

std::unordered_set<uint64_t> ModelInstanceState::pollStopSignals()
{
    auto stoppedReqIds = mWorkItemsQueue.getStoppedReqIds();

    // Merge cancelled requests into stopped requests Ids
    auto cancelledReqIds = mWorkItemsQueue.getCancelledInProgressReqIds();
    stoppedReqIds.insert(cancelledReqIds.begin(), cancelledReqIds.end());

    int64_t nStoppedReqIds = static_cast<int64_t>(stoppedReqIds.size());

    if (getCommWorldSize() > 1)
    {
        // Broadcast number of stopped requests
        bcast(&nStoppedReqIds, 1, MPI_TYPE_INT64_T, 0, COMM_WORLD);

        if (nStoppedReqIds > 0)
        {
            // Broadcast stopped requests Ids
            if (getCommWorldRank() == 0)
            {
                // Store the requestIds in a contiguous vector
                std::vector<uint64_t> stoppedReqIdsVec(stoppedReqIds.begin(), stoppedReqIds.end());
                bcast(stoppedReqIdsVec.data(), stoppedReqIdsVec.size(), MPI_TYPE_UINT64_T, 0, COMM_WORLD);
            }
            else
            {
                std::vector<uint64_t> stoppedReqIdsVec(nStoppedReqIds);
                bcast(stoppedReqIdsVec.data(), stoppedReqIdsVec.size(), MPI_TYPE_UINT64_T, 0, COMM_WORLD);
                // Store the requestIds in the set
                stoppedReqIds.clear();
                std::copy(stoppedReqIdsVec.begin(), stoppedReqIdsVec.end(),
                    std::inserter(stoppedReqIds, stoppedReqIds.end()));
            }
        }
    }
    return stoppedReqIds;
}

void ModelInstanceState::logStats(const std::string& s)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, s.c_str());
#ifdef TRITON_ENABLE_METRICS
    LOG_IF_ERROR(model_state_->UpdateMetrics(s), "Failed updating metrics");
#endif
}

TRITONSERVER_Error* ModelInstanceState::sendTritonResponse(std::shared_ptr<WorkItem> workItem,
    std::list<NamedTensor> const& response_tensors, bool final_response, const std::string& errMsg)
{
    TRITONBACKEND_ResponseFactory* response_factory;
    response_factory = workItem->response_factory();

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));

    auto requestId = workItem->requestId();
    if (final_response)
    {
        mWorkItemsQueue.markFinished(requestId);
    }

    // Check if error
    TRITONSERVER_Error* err = nullptr;
    if (!errMsg.empty())
    {
        std::string errStr = "Encountered error for requestId " + std::to_string(requestId) + ": " + errMsg;
        TLLM_LOG_ERROR(errStr);

        bool is_cancelled = false;
        TRITONBACKEND_ResponseFactoryIsCancelled(response_factory, &is_cancelled);

        auto err_code = is_cancelled ? TRITONSERVER_ERROR_CANCELLED : TRITONSERVER_ERROR_INTERNAL;

        err = TRITONSERVER_ErrorNew(err_code, errStr.c_str());
        final_response = true;
    }
    else
    {
        for (auto it = response_tensors.begin(); it != response_tensors.end(); ++it)
        {
            auto tensor = *it;
            if (!workItem->hasOutputName(tensor.name))
            {
                continue;
            }
            auto shape = tensor.tensor->getShape(); // returns std::vectorint64_t>
            std::vector<int64_t> vshape(shape.nbDims);
            for (std::size_t i = 0; i < vshape.size(); ++i)
            {
                vshape[i] = shape.d[i];
            }

            TRITONBACKEND_Output* output;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, tensor.name.c_str(),
                utils::to_triton_datatype(tensor.tensor->getDataType()), vshape.data(), shape.nbDims));

            uint64_t buffersize = tensor.tensor->getSizeInBytes();
            void* buffer = 0L;
            TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, buffersize, &memory_type, &memory_type_id));
            if (memory_type != TRITONSERVER_MEMORY_CPU && memory_type != TRITONSERVER_MEMORY_CPU_PINNED)
            {
                std::string errStr = "Triton failed to allocate output buffer on CPU";
                err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
                break;
            }
            std::memcpy(buffer, tensor.tensor->data(), buffersize);
        }
    }

    RETURN_IF_ERROR(
        TRITONBACKEND_ResponseSend(response, final_response ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, err));

    return nullptr;
}

} // namespace triton::backend::inflight_batcher_llm
