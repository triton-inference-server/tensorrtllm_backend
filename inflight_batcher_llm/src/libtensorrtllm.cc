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
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <thread>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/NamedTensor.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <nlohmann/json.hpp>
using namespace ::triton::common; // TritonJson

#ifdef TRITON_ENABLE_METRICS
#include "metrics/triton_metrics.h"
#endif
//
// Mockup of LLM inflight batcher based on triton 'minimal' backend example
//

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::mpi;
using namespace std::placeholders; // for _1, _2 etc.

// template class inflight_batcher::batch_manager::GPTManager<float>;

namespace triton
{
namespace backend
{
namespace inflight_batcher_llm
{

inline static const std::string kStopInputTensorName = "stop";
inline static const std::string kStreamingInputTensorName = "streaming";

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

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState
{
public:
    static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);

    template <typename T>
    T GetParameter(const std::string& name)
    {
        assert(false);
    }

    virtual ~ModelState() = default;

#ifdef TRITON_ENABLE_METRICS
    TRITONSERVER_Error* InitMetrics(const std::string& model_name, const uint64_t version, const bool is_v1_model);
    TRITONSERVER_Error* UpdateMetrics(const std::string& statistics);
#endif
    common::TritonJson::Value& GetModelConfig();

private:
#ifdef TRITON_ENABLE_METRICS
    std::unique_ptr<triton_metrics::TritonMetrics> triton_metrics_;
#endif
    common::TritonJson::Value model_config_;
    std::shared_ptr<nvinfer1::ILogger> mTrtLogger{};

    ModelState(TRITONBACKEND_Model* triton_model, TritonJson::Value&& model_config)
        : model_config_(std::move(model_config))
    {
        mTrtLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
        initTrtLlmPlugins(mTrtLogger.get());
#ifdef TRITON_ENABLE_METRICS
        triton_metrics_ = std::make_unique<triton_metrics::TritonMetrics>();
#endif
    }
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
    TRITONSERVER_Message* config_message;
    RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1 /* config_version */, &config_message));

    // We can get the model configuration as a json string from
    // config_message, parse it with our favorite json parser to create
    // DOM that we can access when we need to example the
    // configuration. We use TritonJson, which is a wrapper that returns
    // nice errors (currently the underlying implementation is
    // rapidjson... but others could be added). You can use any json
    // parser you prefer.
    const char* buffer;
    size_t byte_size;
    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

    common::TritonJson::Value model_config;
    TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
    RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
    RETURN_IF_ERROR(err);

    try
    {
        *state = new ModelState(triton_model, std::move(model_config));
    }
    catch (const std::exception& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

common::TritonJson::Value& ModelState::GetModelConfig()
{
    return model_config_;
}

template <>
std::string ModelState::GetParameter<std::string>(const std::string& name)
{
    TritonJson::Value parameters;
    TRITONSERVER_Error* err = model_config_.MemberAsObject("parameters", &parameters);
    if (err != nullptr)
    {
        throw std::runtime_error("Model config doesn't have a parameters section");
        TRITONSERVER_ErrorDelete(err);
    }
    TritonJson::Value value;
    std::string str_value;
    err = parameters.MemberAsObject(name.c_str(), &value);
    if (err != nullptr)
    {
        std::string errStr = "Cannot find parameter with name: " + name;
        throw std::runtime_error(errStr);
        TRITONSERVER_ErrorDelete(err);
    }
    value.MemberAsString("string_value", &str_value);
    return str_value;
}

template <>
int32_t ModelState::GetParameter<int32_t>(const std::string& name)
{
    return std::stoi(GetParameter<std::string>(name));
}

template <>
uint32_t ModelState::GetParameter<uint32_t>(const std::string& name)
{
    return (uint32_t) std::stoul(GetParameter<std::string>(name));
}

template <>
int64_t ModelState::GetParameter<int64_t>(const std::string& name)
{
    return std::stoll(GetParameter<std::string>(name));
}

template <>
uint64_t ModelState::GetParameter<uint64_t>(const std::string& name)
{
    return std::stoull(GetParameter<std::string>(name));
}

template <>
float ModelState::GetParameter<float>(const std::string& name)
{
    return std::stof(GetParameter<std::string>(name));
}

template <>
bool ModelState::GetParameter<bool>(const std::string& name)
{
    auto val = GetParameter<std::string>(name);
    if (val == "True" || val == "true" || val == "TRUE" || val == "1")
    {
        return true;
    }
    else if (val == "False" || val == "false" || val == "FALSE" || val == "0")
    {
        return false;
    }
    else
    {
        std::string err = "Cannot convert " + val + " to a boolean.";
        throw std::runtime_error(err);
    }
}

#ifdef TRITON_ENABLE_METRICS
TRITONSERVER_Error* ModelState::InitMetrics(
    const std::string& model_name, const uint64_t version, const bool is_v1_model)
{
    RETURN_IF_ERROR(triton_metrics_->InitMetrics(model_name, version, is_v1_model));
    return nullptr; // success
}

TRITONSERVER_Error* ModelState::UpdateMetrics(const std::string& statistics)
{
    RETURN_IF_ERROR(triton_metrics_->UpdateMetrics(statistics));
    return nullptr; // success
}
#endif

extern "C"
{

    // Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
    // to allow the backend to create any state associated with the model,
    // and to also examine the model configuration to determine if the
    // configuration is suitable for the backend. Any errors reported by
    // this function will prevent the model from loading.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
    {
        // Create a ModelState object and associate it with the
        // TRITONBACKEND_Model. If anything goes wrong with initialization
        // of the model state then an error is returned and Triton will fail
        // to load the model.
        ModelState* model_state;
        RETURN_IF_ERROR(ModelState::Create(model, &model_state));
        RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

#ifdef TRITON_ENABLE_METRICS
        const char* cname;
        RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
        std::string name(cname);

        uint64_t version;
        RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

        bool is_v1_model = ((model_state->GetParameter<std::string>("gpt_model_type") == "V1")
            || (model_state->GetParameter<std::string>("gpt_model_type") == "v1"));
        LOG_IF_ERROR(model_state->InitMetrics(name, version, is_v1_model), "Failed initializing metrics");
#endif                  // TRITON_ENABLE_METRICS
        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
    // needed. The backend should cleanup any state associated with the
    // model. This function will not be called until all model instances
    // of the model have been finalized.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
    {
        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
        delete model_state;

        return nullptr; // success
    }

} // extern "C"

/////////////

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
public:
    WorkItem(TRITONBACKEND_Request* request, bool isDecoupled)
    {
        uint64_t requestId = (rand() % INT64_MAX) + 1;
        Initialize(request, requestId, isDecoupled);
    }

    WorkItem(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
    {
        Initialize(request, requestId, isDecoupled);
    }

    WorkItem(std::shared_ptr<InferenceRequest> ir, uint64_t RequestId)
        : mInferenceRequest(ir)
        , mRequestId(RequestId)
    {
        factory_ptr_ = nullptr;
    }

    ~WorkItem()
    {
        if (factory_ptr_ != nullptr)
        {
            TRITONBACKEND_ResponseFactoryDelete(factory_ptr_);
        }
    }

    TRITONBACKEND_ResponseFactory* response_factory()
    {
        assert(factory_ptr_ != nullptr);
        return factory_ptr_;
    }

    uint64_t requestId() const
    {
        return mRequestId;
    }

    std::shared_ptr<InferenceRequest> getInferenceRequest() const
    {
        return mInferenceRequest;
    }

    bool hasOutputName(const std::string& outputName)
    {
        return (mRequestOutputNames.find(outputName) != mRequestOutputNames.end());
    }

private:
    void Initialize(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
    {
        mRequestId = requestId;
        mInferenceRequest = createInferenceRequest(request, requestId, isDecoupled);
        mRequestOutputNames = getRequestOutputNames(request);

        // Create response factory for this request
        TRITONBACKEND_ResponseFactoryNew(&factory_ptr_, request);
    }

    // Convert info from original backend request to data structures defined in
    // common/common.h
    std::shared_ptr<InferenceRequest> createInferenceRequest(
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
            TRITONBACKEND_InputProperties(
                input, &input_name, &data_type, &shape, &dims_count, &byte_size, &buffer_count);

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

            NamedTensor t(to_trt_datatype(data_type), shapev, input_name);
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

        bool streamingFlag = getRequestBooleanInputTensor(request, kStreamingInputTensorName);
        inferenceRequest->setIsStreaming(streamingFlag);

        if (streamingFlag && !isDecoupled)
        {
            throw std::runtime_error(
                "Streaming is only supported if model is "
                "deployed using decoupled mode.");
        }

        return inferenceRequest;
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

    std::shared_ptr<InferenceRequest> mInferenceRequest;
    TRITONBACKEND_ResponseFactory* factory_ptr_;
    uint64_t mRequestId;
    std::unordered_set<std::string> mRequestOutputNames;
};

/// @brief Thread-safe queue of work items

class WorkItemsQueue
{
public:
    void clear()
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mPendingWorkItems.clear();
        mPendingWorkItemsReqIds.clear();
        mInProgressWorkItems.clear();
        mStoppedReqIds.clear();
    }

    // Note: this function only be called under a lock
    bool hasInProgressReqId(const uint64_t reqId) const
    {
        return (mInProgressWorkItems.find(reqId) != mInProgressWorkItems.end());
    }

    // Note: this function only be called under a lock
    bool hasPendingReqId(const uint64_t reqId) const
    {
        return (mPendingWorkItemsReqIds.find(reqId) != mPendingWorkItemsReqIds.end());
    }

    /// @brief Add a batch of new work item to the queue
    /// Throws an error if requestId already exists
    std::vector<std::shared_ptr<std::exception>> pushBatch(
        std::vector<std::pair<uint64_t, TRITONBACKEND_Request*>>& requestsToPush, bool isDecoupled)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        std::vector<std::shared_ptr<std::exception>> reqExceptions;
        for (auto& [requestId, request] : requestsToPush)
        {
            if (requestId != 0 && (hasInProgressReqId(requestId) || hasPendingReqId(requestId)))
            {
                std::string errStr
                    = "requestId " + std::to_string(requestId) + " is already in progress, request is ignored.";
                reqExceptions.emplace_back(std::make_shared<std::runtime_error>(errStr));
            }
            else
            {
                auto workItem = requestId != 0 ? std::make_shared<WorkItem>(request, requestId, isDecoupled)
                                               : std::make_shared<WorkItem>(request, isDecoupled);
                mPendingWorkItems.push_back(workItem);
                mPendingWorkItemsReqIds.insert(workItem->requestId());
                reqExceptions.push_back(nullptr);
            }
        }
        return reqExceptions;
    }

    /// @brief Add a new work item to the queue
    /// Throws an error if requestId already exists
    void push(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (hasInProgressReqId(requestId) || hasPendingReqId(requestId))
        {
            std::string errStr
                = "requestId " + std::to_string(requestId) + " is already in progress, request is ignored.";
            throw std::runtime_error(errStr);
        }
        else
        {
            auto workItem = std::make_shared<WorkItem>(request, requestId, isDecoupled);
            mPendingWorkItems.push_back(workItem);
            mPendingWorkItemsReqIds.insert(workItem->requestId());
        }
    }

    void push(TRITONBACKEND_Request* request, bool isDecoupled)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        auto workItem = std::make_shared<WorkItem>(request, isDecoupled);
        mPendingWorkItems.push_back(workItem);
        mPendingWorkItemsReqIds.insert(workItem->requestId());
    }

    /// @brief Get a new work item from the queue, and move it to the list of
    /// in progress work items if it hasn't been stopped
    /// @return A tuple of the workItem and a boolean flag indicating if the work
    /// item has been marked in progress
    /// In case the queue is empty, return nullptr

    std::tuple<std::shared_ptr<WorkItem>, bool> pop()
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mPendingWorkItems.empty())
        {
            return {nullptr, false};
        }

        auto workItem = mPendingWorkItems.front();
        mPendingWorkItems.pop_front();
        mPendingWorkItemsReqIds.erase(workItem->requestId());

        // Check if work item has been stopped
        bool is_stopped = mStoppedReqIds.count(workItem->requestId());

        // Check if the Triton request has been cancelled
        bool is_cancelled = false;
        TRITONBACKEND_ResponseFactoryIsCancelled(workItem->response_factory(), &is_cancelled);

        bool stoppedRequest = false;
        if (!is_stopped && !is_cancelled)
        {
            mInProgressWorkItems.emplace(std::make_pair(workItem->requestId(), workItem));
        }
        else
        {
            mStoppedReqIds.erase(workItem->requestId());
            stoppedRequest = true;
        }

        return {workItem, stoppedRequest};
    }

    size_t numPendingWorkItems() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mPendingWorkItems.size();
    }

    std::shared_ptr<WorkItem> getInProgressWorkItem(uint64_t requestId)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mInProgressWorkItems.at(requestId);
    }

    /// @brief  Mark a request as being finished
    /// @param requestId
    void markFinished(const uint64_t requestId)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (hasInProgressReqId(requestId))
        {
            mInProgressWorkItems.erase(requestId);
        }

        if (mStoppedReqIds.find(requestId) != mStoppedReqIds.end())
        {
            mStoppedReqIds.erase(requestId);
        }
    }

    // Stop a request by adding the request Id to a set
    // The set of stopped request id is used by the poll callback
    // and the pop function
    void stopWorkItem(const uint64_t requestId)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        TLLM_LOG_DEBUG("Stopping request");
        if (hasInProgressReqId(requestId) || hasPendingReqId(requestId))
        {
            mStoppedReqIds.emplace(requestId);
        }
        else
        {
            std::string errStr = std::string("Received stop request for requestId ") + std::to_string(requestId)
                + std::string(" but it's not active (might be completed already).");
            throw std::runtime_error(errStr);
        }
    }

    std::unordered_set<uint64_t> getStoppedReqIds() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mStoppedReqIds;
    }

    std::unordered_set<uint64_t> getCancelledInProgressReqIds() const
    {
        std::unordered_set<uint64_t> cancelledInProgressReqIds;
        {
            std::lock_guard<std::mutex> lk(mMutex);
            for (const auto& pair : mInProgressWorkItems)
            {
                bool is_cancelled = false;
                TRITONBACKEND_ResponseFactoryIsCancelled(pair.second->response_factory(), &is_cancelled);
                if (is_cancelled)
                {
                    cancelledInProgressReqIds.emplace(pair.first);
                }
            }
        }
        return cancelledInProgressReqIds;
    }

private:
    /// Queue of work items
    std::list<std::shared_ptr<WorkItem>> mPendingWorkItems;
    /// requestIds of work items in the queue
    std::set<uint64_t> mPendingWorkItemsReqIds;

    /// work items currently in progress
    std::unordered_map<uint64_t, std::shared_ptr<WorkItem>> mInProgressWorkItems;

    /// ids of the work items that have been stopped
    std::unordered_set<uint64_t> mStoppedReqIds;

    mutable std::mutex mMutex;
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
//
class ModelInstanceState
{
public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state);

    virtual ~ModelInstanceState()
    {
        // terminate decoupled execution loop
        {
            mWorkItemsQueue.clear();
        }
    }

    // Get the state of the model that corresponds to this instance.
    ModelState* StateForModel() const
    {
        return model_state_;
    }

    bool isDecoupled() const
    {
        return mIsDecoupled;
    }

    uint64_t getRequestId(TRITONBACKEND_Request* request)
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
                    std::string err = std::string("Invalid requestId, must be uint64_t. Got ") + strRequestId;
                    throw std::runtime_error(err);
                }
            }
        }

        return requestId;
    }

    // For stop requests, or in case of error during enqueue, we need to send a
    // response to the client
    void sendEnqueueResponse(TRITONBACKEND_Request* request, const std::string& errMsg = "")
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

    void enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count, bool isDecoupled)
    {
        std::vector<std::pair<uint64_t, TRITONBACKEND_Request*>> requestsToPush;

        for (uint32_t r = 0; r < request_count; ++r)
        {
            TRITONBACKEND_Request* request = requests[r];
            try
            {
                auto requestId = getRequestId(request);
                bool stopRequest = getRequestBooleanInputTensor(request, kStopInputTensorName);

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
    std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(const int max_num_requests)
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
            for (size_t i = 0; i < numPendingWorkItems && rval.size() < max_num_requests; ++i)
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
                        packed.insert(
                            packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
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

    TRITONSERVER_Error* sendTritonResponse(std::shared_ptr<WorkItem> workItem,
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
                for (int i = 0; i < vshape.size(); ++i)
                {
                    vshape[i] = shape.d[i];
                }

                TRITONBACKEND_Output* output;
                RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, tensor.name.c_str(),
                    to_triton_datatype(tensor.tensor->getDataType()), vshape.data(), shape.nbDims));

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

    void sendResponse(uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response,
        const std::string& errMsg)
    {
        if (getCommWorldRank() == 0)
        {
            std::string errStr
                = std::string("Failed to send Triton response for requestId: ") + std::to_string(requestId);
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

    std::unordered_set<uint64_t> pollStopSignals()
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

    void logStats(const std::string& s)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, s.c_str());
#ifdef TRITON_ENABLE_METRICS
        LOG_IF_ERROR(model_state_->UpdateMetrics(s), "Failed updating metrics");
#endif
    }

private:
    ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
        : model_state_(model_state)
        , mIsDecoupled(false)
    {
        // Note: std::string::compare fails this test (always return non-zero
        // value). Using old school strcmp instead.
        if (model_state_->GetParameter<std::string>("gpt_model_type") == "V1"
            || model_state_->GetParameter<std::string>("gpt_model_type") == "v1")
        {
            mTrtGptModelType = TrtGptModelType::V1;
        }
        else if (model_state_->GetParameter<std::string>("gpt_model_type") == "inflight_batching")
        {
            mTrtGptModelType = TrtGptModelType::InflightBatching;
        }
        else if (model_state_->GetParameter<std::string>("gpt_model_type") == "inflight_fused_batching")
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

        auto schedulerPolicy = batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT;
        try
        {
            std::string schedulerPolicyStr = model_state_->GetParameter<std::string>("batch_scheduler_policy");
            if (schedulerPolicyStr == "max_utilization")
            {
                schedulerPolicy = batch_scheduler::SchedulerPolicy::MAX_UTILIZATION;
            }
            else if (schedulerPolicyStr == "guaranteed_no_evict")
            {
                schedulerPolicy = batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT;
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

        if (mIsDecoupled && schedulerPolicy != batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT)
        {
            TLLM_LOG_WARNING(
                "The batch scheduler policy will be set to guaranteed_no_evict"
                "since the backend operates in decoupled mode");
            schedulerPolicy = batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT;
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
                const std::string& errMsg)
            { return sendResponse(requestId, response_tensors, final_response, errMsg); },
            [this]() { return pollStopSignals(); }, [this](const std::string& s) { return logStats(s); },
            optionalParams, std::nullopt, std::nullopt, excludeInputInOutput);

        if (getCommWorldRank() != 0)
        {
            while (true)
            {
            }
        }
    }

    ModelState* model_state_;

    //
    // inflight batcher is a decoupled design.
    // It uses response factory objects to decouple responses from requests.
    //
    // New requests are added to mWorkItems list. This list is processed
    // in an infinite loop run by a worker thread. Requests take multiple
    // iterations to complete, and number of iterations is not known in
    // advance. To facilitate this, we use response factory objects to
    // decouple requests and responses.
    //
    TrtGptModelType mTrtGptModelType;
    std::string mModelPath;
    bool mIsDecoupled;

    std::shared_ptr<GptManager> mBatchManager;

    WorkItemsQueue mWorkItemsQueue;
};

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

extern "C"
{

    // Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
    // instance is created to allow the backend to initialize any state
    // associated with the instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
    {
        // Get the model state associated with this instance's model.
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        void* vmodelstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

        // Create a ModelInstanceState object and associate it with the
        // TRITONBACKEND_ModelInstance.
        ModelInstanceState* instance_state;
        RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
    // instance is no longer needed. The backend should cleanup any state
    // associated with the model instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
    {
        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
        ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);
        delete instance_state;

        return nullptr; // success
    }

} // extern "C"

/////////////

extern "C"
{

    // When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
    // that a backend create a response for each request in the batch. A
    // response may be the output tensors required for that request or may
    // be an error that is returned in the response.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
        TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
    {
        ModelInstanceState* instance_state;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

        auto isDecoupled = instance_state->isDecoupled();

        instance_state->enqueue(requests, request_count, isDecoupled);

        for (uint32_t r = 0; r < request_count; ++r)
        {
            TRITONBACKEND_Request* request = requests[r];
            TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
        }

        return nullptr; // success
    }

} // extern "C"

} // namespace inflight_batcher_llm
} // namespace backend
} // namespace triton
