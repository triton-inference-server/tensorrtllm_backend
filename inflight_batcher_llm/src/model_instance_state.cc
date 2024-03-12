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

#include "model_instance_state.h"

#include "tensorrt_llm/common/mpiUtils.h"

namespace mpi = tensorrt_llm::mpi;

namespace triton::backend::inflight_batcher_llm
{

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state)
{
    try
    {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    }
    catch (std::exception const& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelInstanceState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : model_state_(model_state)
    , modelInstance_(triton_model_instance)
    , mHasActiveRequests(false)
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

#ifdef TRITON_ENABLE_METRICS
    custom_metrics_reporter_ = std::make_unique<custom_metrics_reporter::CustomMetricsReporter>();
    custom_metrics_reporter_->InitializeReporter(
        model_state->GetModelName(), model_state->GetModelVersion(), (mTrtGptModelType == TrtGptModelType::V1));
#endif

    mWorkItemsQueue = std::make_unique<WorkItemsQueue>(isDecoupled());

    // Note: std::string::compare fails this test (always return non-zero
    // value). Using old school strcmp instead.
    mModelPath = model_state_->GetParameter<std::string>("gpt_model_path");
    auto configPath = mModelPath + "/config.json";
    std::ifstream jsonStream(configPath);
    TLLM_CHECK_WITH_INFO(jsonStream.is_open(), "Cannot find engine config file %s", configPath.c_str());

    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);

    int32_t maxBeamWidth = 1;
    try
    {
        maxBeamWidth = model_state_->GetParameter<int32_t>("max_beam_width");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("max_beam_width is not specified, will use default value of 1");
    }

    std::optional<int32_t> maxTokensInPagedKvCache = std::nullopt;
    try
    {
        maxTokensInPagedKvCache = model_state_->GetParameter<int32_t>("max_tokens_in_paged_kv_cache");
    }
    catch (std::exception const& e)
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
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(e.what());
    }

    bool enableChunkedContext = false;
    try
    {
        enableChunkedContext = model_state_->GetParameter<bool>("enable_chunked_context");
        if (enableChunkedContext)
        {
            TLLM_LOG_WARNING(
                "enable_chunked_context is set to true, will use context chunking "
                "(requires building the model with use_paged_context_fmha).");
        }
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_chunked_context is not specified, will be set to false.");
    }

    if (isDecoupled() && schedulerPolicy != SchedulerPolicy::GUARANTEED_NO_EVICT)
    {
        if (!enableChunkedContext)
        {
            TLLM_LOG_WARNING(
                "Decoupled mode with a batch scheduler policy other than guaranteed_no_evict "
                "requires building the model with use_paged_context_fmha and setting "
                "enable_chunked_context to true. "
                "The batch scheduler policy will be set to guaranteed_no_evict "
                "since enable_chunked_context is false.");
            schedulerPolicy = SchedulerPolicy::GUARANTEED_NO_EVICT;
        }
    }

    std::optional<float> kvCacheFreeGpuMemFraction = std::nullopt;
    try
    {
        kvCacheFreeGpuMemFraction = model_state_->GetParameter<float>("kv_cache_free_gpu_mem_fraction");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "kv_cache_free_gpu_mem_fraction is not specified, will use default value of 0.9 or "
            "max_tokens_in_paged_kv_cache");
    }

    bool enableTrtOverlap = false;
    try
    {
        enableTrtOverlap = model_state_->GetParameter<bool>("enable_trt_overlap");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_trt_overlap is not specified, will be set to false");
    }

    bool normalizeLogProbs = true;
    try
    {
        normalizeLogProbs = model_state_->GetParameter<bool>("normalize_log_probs");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("normalize_log_probs is not specified, will be set to true");
    }

    bool excludeInputInOutput = false;
    try
    {
        excludeInputInOutput = model_state_->GetParameter<bool>("exclude_input_in_output");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("exclude_input_in_output is not specified, will be set to false");
    }

    std::optional<int32_t> maxAttentionWindow = std::nullopt;
    try
    {
        maxAttentionWindow = model_state_->GetParameter<int32_t>("max_attention_window_size");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "max_attention_window_size is not specified, will "
            "use default value (i.e. max_sequence_length)");
    }

    bool enableKVCacheReuse = false;
    try
    {
        enableKVCacheReuse = model_state_->GetParameter<bool>("enable_kv_cache_reuse");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_kv_cache_reuse is not specified, will be set to false");
    }

    std::optional<DecodingMode> decodingMode = std::nullopt;
    try
    {
        std::string decodingModeStr = model_state_->GetParameter<std::string>("decoding_mode");
        if (decodingModeStr == "top_k")
        {
            decodingMode = DecodingMode::TopK();
        }
        else if (decodingModeStr == "top_p")
        {
            decodingMode = DecodingMode::TopP();
        }
        else if (decodingModeStr == "top_k_top_p")
        {
            decodingMode = DecodingMode::TopKTopP();
        }
        else if (decodingModeStr == "beam_search")
        {
            decodingMode = DecodingMode::BeamSearch();
        }
        else
        {
            throw std::runtime_error("");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(
            "decoding_mode parameter is invalid or not specified"
            "(must be one of the {top_k, top_p, top_k_top_p, beam_search})."
            "Using default: top_k_top_p if max_beam_width == 1, beam_search otherwise");
    }

    auto const gpuDeviceIds = model_state_->GetDeviceIds();

    TrtGptModelOptionalParams optionalParams;
    optionalParams.kvCacheConfig.maxTokens = maxTokensInPagedKvCache;
    optionalParams.kvCacheConfig.freeGpuMemoryFraction = kvCacheFreeGpuMemFraction;
    optionalParams.kvCacheConfig.maxAttentionWindow = maxAttentionWindow;
    optionalParams.kvCacheConfig.enableBlockReuse = enableKVCacheReuse;
    optionalParams.enableTrtOverlap = enableTrtOverlap;
    optionalParams.normalizeLogProbs = normalizeLogProbs;
    optionalParams.enableChunkedContext = enableChunkedContext;
    optionalParams.deviceIds = gpuDeviceIds;
    optionalParams.decodingMode = decodingMode;

    mBatchManager = std::make_shared<GptManager>(
        mModelPath, mTrtGptModelType, maxBeamWidth, schedulerPolicy,
        [this](int max_num_requests) { return get_inference_requests(max_num_requests); },
        [this](uint64_t requestId, std::list<NamedTensor> response_tensors, bool final_response,
            std::string const& errMsg) { return sendResponse(requestId, response_tensors, final_response, errMsg); },
        [this]() { return pollStopSignals(); }, [this](std::string const& s) { return logStats(s); }, optionalParams,
        std::nullopt, std::nullopt, excludeInputInOutput);

    if (COMM_SESSION.getRank() != 0)
    {
        while (true)
        {
        }
    }
}

void ModelInstanceState::enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count)
{
    std::vector<WorkItemsQueue::RequestWrapper> requestsToPush;
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    for (uint32_t r = 0; r < request_count; ++r)
    {
        TRITONBACKEND_Request* request = requests[r];
        try
        {
            auto requestId = utils::getRequestId(request, mRequestIdStrMap);
            bool stopRequest = utils::getRequestBooleanInputTensor(request, kStopInputTensorName);

            if (stopRequest)
            {
                if (requestId != 0)
                {
                    // Check if request is in progress or in queue, if not ignore
                    mWorkItemsQueue->stopWorkItem(requestId);
                    // Send a response back to client for stop request
                    utils::sendEnqueueResponse(request);
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
        catch (std::exception const& e)
        {
            // In case of error, no work item is added to queue, so response
            // callback needs to be called
            utils::sendEnqueueResponse(request, e.what());
        }
    }

    auto exceptions = mWorkItemsQueue->pushBatch(requestsToPush, exec_start_ns);

    for (uint32_t r = 0; r < requestsToPush.size(); ++r)
    {
        auto request = requestsToPush.at(r).triton_request;
        auto e = exceptions.at(r);
        if (e)
        {
            utils::sendEnqueueResponse(request, e->what());
        }
    }

    return;
}

// Return up to max_num_requests inference requests.
std::list<std::shared_ptr<InferenceRequest>> ModelInstanceState::get_inference_requests(int const max_num_requests)
{
    std::list<std::shared_ptr<InferenceRequest>> rval;
    if (max_num_requests <= 0)
    {
        return rval;
    }

    auto const& commSession = COMM_SESSION;

    auto world_size = commSession.getSize();
    auto rank = commSession.getRank();
    if (rank == 0)
    {
        auto numPendingWorkItems = mWorkItemsQueue->numPendingWorkItems();
        // Loop over the pending work items and include at most `max_num_requests`
        for (size_t i = 0; i < numPendingWorkItems && static_cast<int>(rval.size()) < max_num_requests; ++i)
        {
            auto [workItem, stoppedRequest] = mWorkItemsQueue->pop();

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
            mHasActiveRequests = (num_new_work_items > 0 || mBatchManager->getNumActiveRequests() > 0);
            if (mHasActiveRequests)
            {
                commSession.bcastValue(num_new_work_items, 0);
            }

            if (num_new_work_items > 0)
            {
                std::vector<int64_t> packed;
                for (auto ir : rval)
                {
                    auto vpacked = ir->serialize();
                    packed.push_back(static_cast<int64_t>(vpacked.size()));
                    packed.insert(packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                }
                commSession.bcast(packed, 0);
            }
        }
    }
    else
    {
        // subordinate ranks hang until master rank sends work
        int64_t num_new_work_items;
        commSession.bcastValue(num_new_work_items, 0);
        mHasActiveRequests = (num_new_work_items > 0 || mBatchManager->getNumActiveRequests() > 0);
        if (num_new_work_items > 0)
        {
            std::vector<int64_t> packed;
            commSession.bcast(packed, 0);
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
    uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response, std::string const& errMsg)
{
    if (COMM_SESSION.getRank() == 0)
    {
        std::string errStr = std::string("Failed to send Triton response for requestId: ")
            + utils::getRequestIdStr(requestId, mRequestIdStrMap);
        if (final_response)
        {
            mRequestIdStrMap.erase(requestId);
        }
        try
        {
            auto workItem = mWorkItemsQueue->getInProgressWorkItem(requestId);
            auto tritonErr = sendTritonResponse(workItem, response_tensors, final_response, errMsg);
            LOG_IF_ERROR(tritonErr, errStr);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(errStr);
        }
    }
}

std::unordered_set<uint64_t> ModelInstanceState::pollStopSignals()
{
    auto stoppedReqIds = mWorkItemsQueue->getStoppedReqIds();

    // Merge cancelled requests into stopped requests Ids
    auto cancelledReqIds = mWorkItemsQueue->getCancelledInProgressReqIds();
    stoppedReqIds.insert(cancelledReqIds.begin(), cancelledReqIds.end());

    int64_t nStoppedReqIds = static_cast<int64_t>(stoppedReqIds.size());

    auto const& commSession = COMM_SESSION;

    if (commSession.getSize() > 1 && mHasActiveRequests)
    {
        // Broadcast number of stopped requests
        commSession.bcastValue(nStoppedReqIds, 0);

        if (nStoppedReqIds > 0)
        {
            // Broadcast stopped requests Ids
            if (commSession.getRank() == 0)
            {
                // Store the requestIds in a contiguous vector
                std::vector<uint64_t> stoppedReqIdsVec(stoppedReqIds.begin(), stoppedReqIds.end());
                commSession.bcast(stoppedReqIdsVec.data(), stoppedReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
            }
            else
            {
                std::vector<uint64_t> stoppedReqIdsVec(nStoppedReqIds);
                commSession.bcast(stoppedReqIdsVec.data(), stoppedReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
                // Store the requestIds in the set
                stoppedReqIds.clear();
                std::copy(stoppedReqIdsVec.begin(), stoppedReqIdsVec.end(),
                    std::inserter(stoppedReqIds, stoppedReqIds.end()));
            }
        }
    }

    return stoppedReqIds;
}

void ModelInstanceState::logStats(std::string const& s)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, s.c_str());
#ifdef TRITON_ENABLE_METRICS
    LOG_IF_ERROR(custom_metrics_reporter_->UpdateCustomMetrics(s), "Failed updating TRT LLM statistics");
#endif
}

TRITONSERVER_Error* ModelInstanceState::sendTritonResponse(std::shared_ptr<WorkItem> workItem,
    std::list<NamedTensor> const& response_tensors, bool final_response, std::string const& errMsg)
{
    TRITONBACKEND_ResponseFactory* response_factory;
    response_factory = workItem->response_factory();

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));

    auto requestId = workItem->requestId();
    if (final_response)
    {
        SET_TIMESTAMP(workItem->getTimestamps().compute_end_ns);
        mWorkItemsQueue->markFinished(requestId);
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

    if (final_response)
    {
        LOG_IF_ERROR(workItem->reportBaseMetrics(modelInstance_, err), "Error reporting base metrics");
        // Reporting Triton core metrics requires the use of the original TRITONBACKEND_Request.
        // Therefore we hold off releasing the request until this point.
        TRITONBACKEND_RequestRelease(workItem->getTritonInferenceRequest(), TRITONSERVER_REQUEST_RELEASE_ALL);
    }

    RETURN_IF_ERROR(
        TRITONBACKEND_ResponseSend(response, final_response ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, err));

    return nullptr;
}

} // namespace triton::backend::inflight_batcher_llm
