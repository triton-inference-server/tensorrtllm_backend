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
#include "utils.h"

#include <nlohmann/json.hpp>

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

executor::BatchingType ModelInstanceState::getBatchingTypeFromParams()
{
    executor::BatchingType batchingType;
    auto gpt_model_type = model_state_->GetParameter<std::string>("gpt_model_type");

    if (gpt_model_type == "V1" || gpt_model_type == "v1")
    {
        batchingType = executor::BatchingType::kSTATIC;
    }
    else if (gpt_model_type == "inflight_batching" || gpt_model_type == "inflight_fused_batching")
    {
        batchingType = executor::BatchingType::kINFLIGHT;
    }
    else
    {
        throw std::runtime_error(
            "Invalid gpt_model_type. Must be "
            "v1/inflight_batching/inflight_fused_batching.");
    }
    return batchingType;
}

executor::KvCacheConfig ModelInstanceState::getKvCacheConfigFromParams()
{
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

    std::optional<size_t> kvCacheHostCacheSize = std::nullopt;
    try
    {
        kvCacheHostCacheSize = model_state_->GetParameter<size_t>("kv_cache_host_memory_bytes");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("kv_cache_host_memory_bytes not set, defaulting to 0");
    }

    bool kvCacheOnboardBlocks = true;
    try
    {
        kvCacheOnboardBlocks = model_state_->GetParameter<bool>("kv_cache_onboard_blocks");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("kv_cache_onboard_blocks not set, defaulting to true");
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

    std::optional<int32_t> sinkTokenLength = std::nullopt;
    try
    {
        sinkTokenLength = model_state_->GetParameter<int32_t>("sink_token_length");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "sink_token_length is not specified, will "
            "use default value");
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

    std::optional<SizeType32> maxAttentionWindowSizeType = std::nullopt;
    if (maxAttentionWindow.has_value())
    {
        maxAttentionWindowSizeType = static_cast<SizeType32>(maxAttentionWindow.value());
    }

    return executor::KvCacheConfig(enableKVCacheReuse, maxTokensInPagedKvCache, maxAttentionWindowSizeType,
        sinkTokenLength, kvCacheFreeGpuMemFraction, kvCacheHostCacheSize, kvCacheOnboardBlocks);
}

executor::ParallelConfig ModelInstanceState::getParallelConfigFromParams()
{
    executor::ParallelConfig parallelConfig;
    auto const gpuDeviceIds = model_state_->GetDeviceIds();
    if (gpuDeviceIds.has_value())
    {
        parallelConfig.setDeviceIds(gpuDeviceIds.value());
    }

    char const* str = std::getenv("TRTLLM_ORCHESTRATOR");
    if (str && std::atoi(str) != 0)
    {
        parallelConfig.setCommunicationMode(executor::CommunicationMode::kORCHESTRATOR);
        auto workerExecutablePath = model_state_->GetExecutorWorkerPath();
        auto orchestratorConfig = executor::OrchestratorConfig(true, workerExecutablePath);
        parallelConfig.setOrchestratorConfig(orchestratorConfig);
    }
    return parallelConfig;
}

executor::PeftCacheConfig ModelInstanceState::getPeftCacheConfigFromParams()
{
    // parse LoRA / Peft cache parameters
    // lora_cache_max_adapter_size
    // lora_cache_optimal_adapter_size
    // lora_cache_gpu_memory_fraction
    // lora_cache_host_memory_bytes

    SizeType32 maxAdapterSize = 64;
    SizeType32 optimalAdapterSize = 8;
    std::optional<size_t> hostCacheSize = std::nullopt;
    std::optional<float> deviceCachePercent = std::nullopt;

    std::string fieldName = "lora_cache_max_adapter_size";
    try
    {
        maxAdapterSize = model_state_->GetParameter<SizeType32>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 64");
    }

    fieldName = "lora_cache_optimal_adapter_size";
    try
    {
        optimalAdapterSize = model_state_->GetParameter<SizeType32>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 8");
    }
    fieldName = "lora_cache_gpu_memory_fraction";
    try
    {
        deviceCachePercent = model_state_->GetParameter<float>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 0.05");
    }
    fieldName = "lora_cache_host_memory_bytes";
    try
    {
        hostCacheSize = model_state_->GetParameter<size_t>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 1GB");
    }

    return executor::PeftCacheConfig(0, 0, optimalAdapterSize, maxAdapterSize,
        ModelInstanceState::kPeftCacheNumPutWorkers, ModelInstanceState::kPeftCacheNumEnsureWorkers,
        ModelInstanceState::kPeftCacheNumCopyStreams, 24, 8, deviceCachePercent, hostCacheSize);
}

executor::SchedulerConfig ModelInstanceState::getSchedulerConfigFromParams(bool enableChunkedContext)
{
    using executor::CapacitySchedulerPolicy;
    auto schedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    try
    {
        std::string schedulerPolicyStr = model_state_->GetParameter<std::string>("batch_scheduler_policy");
        if (schedulerPolicyStr == "max_utilization")
        {
            schedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
        }
        else if (schedulerPolicyStr == "guaranteed_no_evict")
        {
            schedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
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

    if (isDecoupled() && schedulerPolicy != CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        if (!enableChunkedContext)
        {
            TLLM_LOG_WARNING(
                "Decoupled mode with a batch scheduler policy other than guaranteed_no_evict "
                "requires building the model with use_paged_context_fmha and setting "
                "enable_chunked_context to true. "
                "The batch scheduler policy will be set to guaranteed_no_evict "
                "since enable_chunked_context is false.");
            schedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
        }
    }
    return executor::SchedulerConfig(schedulerPolicy);
}

executor::ExecutorConfig ModelInstanceState::getExecutorConfigFromParams()
{
    auto batchingType = getBatchingTypeFromParams();

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

    int32_t iterStatsMaxIterations = executor::kDefaultIterStatsMaxIterations;
    try
    {
        iterStatsMaxIterations = model_state_->GetParameter<int32_t>("iter_stats_max_iterations");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("iter_stats_max_iterations is not specified, will use default value of "
            + std::to_string(iterStatsMaxIterations));
    }

    int32_t requestStatsMaxIterations = executor::kDefaultRequestStatsMaxIterations;
    try
    {
        requestStatsMaxIterations = model_state_->GetParameter<int32_t>("request_stats_max_iterations");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("request_stats_max_iterations is not specified, will use default value of "
            + std::to_string(requestStatsMaxIterations));
    }

    try
    {
        model_state_->GetParameter<bool>("enable_trt_overlap");
        TLLM_LOG_WARNING("enable_trt_overlap is deprecated and will be ignored");
    }
    catch (std::exception const& e)
    {
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

    executor::ExecutorConfig executorConfig;

    auto kvCacheConfig = getKvCacheConfigFromParams();

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

    auto schedulerConfig = getSchedulerConfigFromParams(enableChunkedContext);

    auto peftCacheConfig = getPeftCacheConfigFromParams();

    auto parallelConfig = getParallelConfigFromParams();

    std::optional<executor::DecodingMode> decodingMode = std::nullopt;
    try
    {
        std::string decodingModeStr = model_state_->GetParameter<std::string>("decoding_mode");
        if (decodingModeStr == "top_k")
        {
            decodingMode = executor::DecodingMode::kTOP_K;
        }
        else if (decodingModeStr == "top_p")
        {
            decodingMode = executor::DecodingMode::kTOP_P;
        }
        else if (decodingModeStr == "top_k_top_p")
        {
            decodingMode = executor::DecodingMode::kTOP_K_TOP_P;
        }
        else if (decodingModeStr == "beam_search")
        {
            decodingMode = executor::DecodingMode::kBEAM_SEARCH;
        }
        else if (decodingModeStr == "medusa")
        {
            decodingMode = executor::DecodingMode::kMEDUSA;
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

    std::optional<executor::MedusaChoices> medusaChoices = std::nullopt;
    try
    {
        medusaChoices = model_state_->GetParameter<executor::MedusaChoices>("medusa_choices");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(
            "medusa_choices parameter is not specified. "
            "Will be using default mc_sim_7b_63 choices instead");
    }

    float gpuWeightsPercent = 1.0f;
    try
    {
        gpuWeightsPercent = model_state_->GetParameter<float>("gpu_weights_percent");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("gpu_weights_percent parameter is not specified, will use default value of 1.0");
    }

    return executor::ExecutorConfig(maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext,
        normalizeLogProbs, iterStatsMaxIterations, requestStatsMaxIterations, batchingType, parallelConfig,
        peftCacheConfig, std::nullopt, medusaChoices, decodingMode, gpuWeightsPercent);
}

ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : model_state_(model_state)
    , modelInstance_(triton_model_instance)
{
    mModelPath = model_state_->GetParameter<std::string>("gpt_model_path");

    auto executorConfig = getExecutorConfigFromParams();

#ifdef TRITON_ENABLE_METRICS
    custom_metrics_reporter_ = std::make_unique<custom_metrics_reporter::CustomMetricsReporter>();
    custom_metrics_reporter_->InitializeReporter(model_state->GetModelName(), model_state->GetModelVersion(),
        (executorConfig.getBatchingType() == executor::BatchingType::kSTATIC));
#endif

    mExecutor.reset(new executor::Executor(mModelPath, executor::ModelType::kDECODER_ONLY, executorConfig));

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
    mInstanceSpecificConfig.excludeInputFromOutput = excludeInputInOutput;

    int cancellationCheckPeriodMs = 100;
    try
    {
        cancellationCheckPeriodMs = model_state_->GetParameter<int>("cancellation_check_period_ms");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("cancellation_check_period_ms is not specified, will be set to 100 (ms)");
    }
    mInstanceSpecificConfig.cancellationCheckPeriodMs = cancellationCheckPeriodMs;

    int statsCheckPeriodMs = 100;
    try
    {
        statsCheckPeriodMs = model_state_->GetParameter<int>("stats_check_period_ms");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("stats_check_period_ms is not specified, will be set to 100 (ms)");
    }
    mInstanceSpecificConfig.statsCheckPeriodMs = statsCheckPeriodMs;

    if (mExecutor->canEnqueueRequests())
    {
        mStopWaitForResponse = false;
        mWaitForResponseThread = std::thread(&ModelInstanceState::WaitForResponse, this);

        mStopWaitForStats = false;
        mWaitForStatsThread = std::thread(&ModelInstanceState::WaitForStats, this);

        mStopWaitForCancel = false;
        mWaitForCancelThread = std::thread(&ModelInstanceState::WaitForCancel, this);
    }
    else
    {
        // Shutdown the worker ranks which will cause them to wait for leader/orchestrator to terminate
        mExecutor->shutdown();
    }
}

void ModelInstanceState::sendEnqueueResponse(TRITONBACKEND_Request* request, TRITONSERVER_Error* error)
{
    TRITONBACKEND_ResponseFactory* factory;
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");
    TRITONBACKEND_Response* tritonResponse;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
    LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
        "Cannot send response");
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "Cannot release request");
}

bool ModelInstanceState::handleStopRequest(TRITONBACKEND_Request* request, std::string const& tritonRequestId)
{
    bool stopRequest = utils::getRequestBooleanInputTensor(request, kStopInputTensorName);
    if (!stopRequest)
    {
        return false;
    }

    TRITONSERVER_Error* error = nullptr;

    try
    {
        if (tritonRequestId == "")
        {
            throw std::runtime_error("Trying to stop a request but request ID is not provided");
        }
        std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
        if (mTritonRequestIdToRequestId.count(tritonRequestId))
        {
            auto requestId = mTritonRequestIdToRequestId[tritonRequestId];
            mExecutor->cancelRequest(requestId);
        }
    }
    catch (std::exception const& e)
    {
        error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
    }
    // mTritonRequestIdToRequestId.count(tritonRequestId) == false doesn't necessary mean an error since the
    // request to cancel may already be completed.
    // Send an empty response to indicate the request has been successfully cancelled
    sendEnqueueResponse(request, error);
    return true;
}

executor::Request ModelInstanceState::createExecutorRequest(
    TRITONBACKEND_Request* request, bool excludeInputFromOutput, bool isDecoupled)
{
    auto inputsTensors = utils::readInputsTensors(request);
    bool streaming = utils::getRequestBooleanInputTensor(request, kStreamingInputTensorName);
    return utils::createRequestFromInputTensors(inputsTensors, excludeInputFromOutput, isDecoupled, streaming);
}

void ModelInstanceState::enqueue(TRITONBACKEND_Request** requests, uint32_t const request_count)
{

    uint64_t exec_start_ns{0};
    SET_TIMESTAMP(exec_start_ns);

    for (uint32_t i = 0; i < request_count; ++i)
    {
        TRITONBACKEND_Request* request = requests[i];

        try
        {
            char const* charRequestId = nullptr;
            TRITONBACKEND_RequestId(request, &charRequestId);
            std::string tritonRequestId;
            if (charRequestId != nullptr)
            {
                tritonRequestId = charRequestId;
            }

            if (handleStopRequest(request, tritonRequestId))
            {
                continue;
            }

            auto executorRequest
                = createExecutorRequest(request, mInstanceSpecificConfig.excludeInputFromOutput, isDecoupled());

            int64_t inputTokensSize = executorRequest.getInputTokenIds().size();
            executor::SizeType32 beamWidthCopy = executorRequest.getSamplingConfig().getBeamWidth();
            std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
            uint64_t compute_start_ns{0};
            SET_TIMESTAMP(compute_start_ns);
            auto requestId = mExecutor->enqueueRequest(executorRequest);
            if (mRequestIdToRequestData.count(requestId))
            {
                TLLM_LOG_ERROR(
                    "Executor returns a request ID that already exists. This shouldn't happen unless there is "
                    "something "
                    "wrong in TRT-LLM runtime.");
            }
            TRITONBACKEND_ResponseFactory* factory;
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");

            auto requestOutputNames = utils::getRequestOutputNames(request);
            mRequestIdToRequestData.emplace(requestId,
                RequestData{factory, request, tritonRequestId, inputTokensSize, beamWidthCopy,
                    std::move(requestOutputNames), {exec_start_ns, compute_start_ns, 0, 0}});
            if (tritonRequestId != "")
            {
                mTritonRequestIdToRequestId[tritonRequestId] = requestId;
            }
        }
        catch (std::exception const& e)
        {
            sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what()));
        }
    }
    return;
}

TRITONSERVER_Error* ModelInstanceState::reportBaseMetrics(RequestData& requestData, TRITONSERVER_Error* error)
{
    auto& timestamps = requestData.timestamps;
    SET_TIMESTAMP(timestamps.exec_end_ns);

    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(modelInstance_, requestData.tritonRequest, (error == nullptr),
            timestamps.exec_start_ns, timestamps.compute_start_ns, timestamps.compute_end_ns, timestamps.exec_end_ns));

    // For now we will assume a batch size of 1 for each request. This may change in the future but for
    // now it seems that even when requests are dynamically batched together each workItem is associated
    // with its own request object and is handled independently due to the nature of IFB.
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(modelInstance_, 1 /* batch size */,
        timestamps.exec_start_ns, timestamps.compute_start_ns, timestamps.compute_end_ns, timestamps.exec_end_ns));

    return nullptr; // success
}

std::tuple<TRITONBACKEND_Response*, bool, TRITONSERVER_Error*> ModelInstanceState::fillTritonResponse(
    TRITONBACKEND_ResponseFactory* factory, executor::Response const& response, RequestData const& requestData)
{
    TRITONBACKEND_Response* tritonResponse;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");

    TRITONSERVER_Error* error = nullptr;
    bool isFinal = false;
    try
    {
        if (!response.hasError())
        {
            auto const& result = response.getResult();
            isFinal = result.isFinal;
            error = nullptr;
            auto outputIds = result.outputTokenIds;
            std::vector<int32_t> beamLength(outputIds.size());
            int32_t maxBeamLength = -1;
            for (size_t i = 0; i < outputIds.size(); ++i)
            {
                beamLength[i] = outputIds[i].size();
                maxBeamLength = std::max(beamLength[i], maxBeamLength);
            }
            if (maxBeamLength == -1)
            {
                TLLM_LOG_ERROR("Output ids is empty");
                maxBeamLength = 0;
            }
            for (auto& vec : outputIds)
            {
                vec.resize(maxBeamLength, -1);
            }

            if (requestData.outputNames.count(OutputFieldsNames::outputIds) > 0)
            {
                std::vector<int64_t> outputIdsShape{1, static_cast<int64_t>(outputIds.size()), maxBeamLength};
                auto outputIdsType = TRITONSERVER_TYPE_INT32;
                auto outputIdsBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, outputIdsShape, outputIdsType, OutputFieldsNames::outputIds);
                utils::flatten<int32_t>(outputIds, outputIdsBuffer, outputIdsShape);
            }
            else
            {
                TLLM_THROW("%s tensor must be present in list of output tensors", OutputFieldsNames::outputIds);
            }

            if (requestData.outputNames.count(OutputFieldsNames::sequenceLength) > 0)
            {
                std::vector<int64_t> sequenceLengthShape{1, static_cast<int64_t>(outputIds.size())};
                auto sequenceLengthType = TRITONSERVER_TYPE_INT32;
                auto sequenceLengthBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, sequenceLengthShape, sequenceLengthType, OutputFieldsNames::sequenceLength);
                utils::flatten<int32_t>(beamLength, sequenceLengthBuffer, sequenceLengthShape);
            }
            else
            {
                TLLM_THROW("%s tensor must be present in list of output tensors", OutputFieldsNames::sequenceLength);
            }

            if (requestData.outputNames.count(OutputFieldsNames::contextLogits) > 0)
            {
                if (result.contextLogits.has_value())
                {
                    auto contextLogitsShapeOriginal = result.contextLogits.value().getShape();
                    std::vector<int64_t> contextLogitsShape{
                        1, contextLogitsShapeOriginal[0], contextLogitsShapeOriginal[1]};
                    auto contextLogitsType = TRITONSERVER_TYPE_FP32;
                    auto contextLogitsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, contextLogitsShape, contextLogitsType, OutputFieldsNames::contextLogits);
                    utils::flatten<float>(result.contextLogits.value(), contextLogitsBuffer, contextLogitsShape);
                }
                else
                {
                    std::vector<int64_t> contextLogitsShape{1, 1, 1};
                    auto contextLogitsType = TRITONSERVER_TYPE_FP32;
                    auto contextLogitsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, contextLogitsShape, contextLogitsType, OutputFieldsNames::contextLogits);
                    utils::flatten<float>(std::vector<float>{0}, contextLogitsBuffer, contextLogitsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::generationLogits) > 0)
            {
                if (result.generationLogits.has_value())
                {
                    auto generationLogitsShapeOriginal = result.generationLogits.value().getShape();
                    std::vector<int64_t> generationLogitsShape{1, generationLogitsShapeOriginal[0],
                        generationLogitsShapeOriginal[1], generationLogitsShapeOriginal[2]};
                    auto generationLogitsType = TRITONSERVER_TYPE_FP32;
                    auto generationLogitsBuffer = utils::getResponseBuffer<float>(tritonResponse, generationLogitsShape,
                        generationLogitsType, OutputFieldsNames::generationLogits);
                    utils::flatten<float>(
                        result.generationLogits.value(), generationLogitsBuffer, generationLogitsShape);
                }
                else
                {
                    std::vector<int64_t> generationLogitsShape{1, 1, 1, 1};
                    auto generationLogitsType = TRITONSERVER_TYPE_FP32;
                    auto generationLogitsBuffer = utils::getResponseBuffer<float>(tritonResponse, generationLogitsShape,
                        generationLogitsType, OutputFieldsNames::generationLogits);
                    utils::flatten<float>(std::vector<float>{0}, generationLogitsBuffer, generationLogitsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::outputLogProbs) > 0)
            {
                if (result.logProbs.has_value())
                {
                    std::vector<int64_t> outputLogProbsShape{1, static_cast<int64_t>(result.logProbs.value().size()),
                        static_cast<int64_t>(result.logProbs.value()[0].size())};
                    auto outputLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto outputLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, outputLogProbsShape, outputLogProbsType, OutputFieldsNames::outputLogProbs);
                    utils::flatten<float>(result.logProbs.value(), outputLogProbsBuffer, outputLogProbsShape);
                }
                else
                {
                    std::vector<int64_t> outputLogProbsShape{1, 1, requestData.inputTokensSize};
                    auto outputLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto outputLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, outputLogProbsShape, outputLogProbsType, OutputFieldsNames::outputLogProbs);
                    utils::flatten<float>(
                        std::vector<float>(requestData.inputTokensSize), outputLogProbsBuffer, outputLogProbsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::cumLogProbs) > 0)
            {
                if (result.cumLogProbs.has_value())
                {
                    std::vector<int64_t> cumLogProbsShape{1, static_cast<int64_t>(result.cumLogProbs.value().size())};
                    auto cumLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto cumLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, cumLogProbsShape, cumLogProbsType, OutputFieldsNames::cumLogProbs);
                    utils::flatten<float>(result.cumLogProbs.value(), cumLogProbsBuffer, cumLogProbsShape);
                }
                else
                {
                    std::vector<int64_t> cumLogProbsShape{1, 1};
                    auto cumLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto cumLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, cumLogProbsShape, cumLogProbsType, OutputFieldsNames::cumLogProbs);
                    utils::flatten<float>(std::vector<float>{0}, cumLogProbsBuffer, cumLogProbsShape);
                }
            }
        }
        else
        {
            isFinal = true;
            std::string errMsg = "Executor failed process requestId " + std::to_string(response.getRequestId())
                + " due to the following error: " + response.getErrorMsg();
            error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errMsg.c_str());
        }
    }
    catch (std::exception const& e)
    {
        // In case of error while processing response, return response with error
        isFinal = true;
        std::string errMsg = "Error encountered while populating response: " + std::string(e.what());
        error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errMsg.c_str());
    }

    return {tritonResponse, isFinal, error};
}

void ModelInstanceState::WaitForResponse()
{
    while (!mStopWaitForResponse)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = mExecutor->awaitResponses(waitTime);
        uint64_t compute_end_ns{0};
        SET_TIMESTAMP(compute_end_ns);

        for (auto const& response : responses)
        {
            auto requestId = response.getRequestId();
            RequestData requestData;
            {
                std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
                if (!mRequestIdToRequestData.count(requestId))
                {
                    TLLM_LOG_ERROR("Unexpected response for a request ID that is not active");
                    continue;
                }
                requestData = mRequestIdToRequestData[requestId];
            }

            auto factory = requestData.factory;

            auto [tritonResponse, isFinal, error] = fillTritonResponse(factory, response, requestData);

            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(tritonResponse, isFinal ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, error),
                "Cannot send response");

            if (isFinal)
            {
                std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
                if (requestData.tritonRequestId != "")
                {
                    mTritonRequestIdToRequestId.erase(requestData.tritonRequestId);
                }

                requestData.timestamps.compute_end_ns = compute_end_ns;
                LOG_IF_ERROR(reportBaseMetrics(requestData, error), "Error reporting metrics");

                LOG_IF_ERROR(TRITONBACKEND_RequestRelease(requestData.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
                    "Cannot release request");
                LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
                mRequestIdToRequestData.erase(requestId);
            }
        }
    }
}

void ModelInstanceState::WaitForStats()
{
    while (!mStopWaitForStats)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mInstanceSpecificConfig.statsCheckPeriodMs));
        auto stats = mExecutor->getLatestIterationStats();
        for (auto const& stat : stats)
        {
            std::string statJson = "{";
            statJson.append("\"Active Request Count\":" + std::to_string(stat.numActiveRequests) + ",");
            statJson.append("\"Iteration Counter\":" + std::to_string(stat.iter) + ",");
            statJson.append("\"Max Request Count\":" + std::to_string(stat.maxNumActiveRequests) + ",");
            statJson.append("\"Runtime CPU Memory Usage\":" + std::to_string(stat.cpuMemUsage) + ",");
            statJson.append("\"Runtime GPU Memory Usage\":" + std::to_string(stat.gpuMemUsage) + ",");
            statJson.append("\"Runtime Pinned Memory Usage\":" + std::to_string(stat.pinnedMemUsage) + ",");
            statJson.append("\"Timestamp\":" + ("\"" + stat.timestamp + "\"") + ",");

            if (stat.inflightBatchingStats.has_value())
            {
                auto const& modelStats = stat.inflightBatchingStats.value();
                statJson.append("\"Context Requests\":" + std::to_string(modelStats.numContextRequests) + ",");
                statJson.append("\"Generation Requests\":" + std::to_string(modelStats.numGenRequests) + ",");
                statJson.append("\"MicroBatch ID\":" + std::to_string(modelStats.microBatchId) + ",");
                statJson.append("\"Paused Requests\":" + std::to_string(modelStats.numPausedRequests) + ",");
                statJson.append("\"Scheduled Requests\":" + std::to_string(modelStats.numScheduledRequests) + ",");
                statJson.append("\"Total Context Tokens\":" + std::to_string(modelStats.numCtxTokens) + ",");
            }
            else if (stat.staticBatchingStats.has_value())
            {
                auto const& modelStats = stat.staticBatchingStats.value();
                statJson.append("\"Context Requests\":" + std::to_string(modelStats.numContextRequests) + ",");
                statJson.append("\"Scheduled Requests\":" + std::to_string(modelStats.numScheduledRequests) + ",");
                statJson.append("\"Total Context Tokens\":" + std::to_string(modelStats.numCtxTokens) + ",");
                statJson.append("\"Total Generation Tokens\":" + std::to_string(modelStats.numGenTokens) + ",");
                statJson.append("\"Empty Generation Slots\":" + std::to_string(modelStats.emptyGenSlots) + ",");
            }
            else
            {
                TLLM_LOG_ERROR("Missing stats");
                continue;
            }

            if (stat.kvCacheStats.has_value())
            {
                auto const& kvStats = stat.kvCacheStats.value();
                statJson.append("\"Free KV cache blocks\":" + std::to_string(kvStats.freeNumBlocks) + ",");
                statJson.append("\"Max KV cache blocks\":" + std::to_string(kvStats.maxNumBlocks) + ",");
                statJson.append("\"Tokens per KV cache block\":" + std::to_string(kvStats.tokensPerBlock) + ",");
                statJson.append("\"Used KV cache blocks\":" + std::to_string(kvStats.usedNumBlocks) + ",");
            }

            statJson.back() = '}';

            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, statJson.c_str());
#ifdef TRITON_ENABLE_METRICS
            LOG_IF_ERROR(custom_metrics_reporter_->UpdateCustomMetrics(statJson), "Failed updating TRT LLM statistics");
#endif
        }
    }
}

void ModelInstanceState::WaitForCancel()
{
    while (!mStopWaitForCancel)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mInstanceSpecificConfig.cancellationCheckPeriodMs));
        std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
        for (auto const& pair : mRequestIdToRequestData)
        {
            auto const& requestId = pair.first;
            auto const& requestData = pair.second;
            bool isCancelled = false;
            LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryIsCancelled(requestData.factory, &isCancelled),
                "Failed to query factory status");
            if (isCancelled)
            {
                mExecutor->cancelRequest(requestId);
            }
        }
    }
}

} // namespace triton::backend::inflight_batcher_llm
