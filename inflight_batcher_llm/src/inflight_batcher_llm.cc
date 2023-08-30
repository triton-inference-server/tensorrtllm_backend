// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <list>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <cassert>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "tensorrt_llm/batch_manager/Tensor.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <nlohmann/json.hpp>
#include "mpiUtils.h"

using namespace ::triton::common;   // TritonJson

//
// Mockup of LLM inflight batcher based on triton 'minimal' backend example
//

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace std::placeholders; // for _1, _2 etc.
//template class inflight_batcher::batch_manager::GPTManager<float>;

namespace triton { namespace backend { namespace inflight_batcher_llm {

Tensor_t to_common_datatype(TRITONSERVER_DataType data_type)
{
    if (data_type == TRITONSERVER_TYPE_INVALID) {
        return DT_INVALID;
    } else if (data_type == TRITONSERVER_TYPE_BOOL) {
        return DT_BOOL;
    } else if (data_type == TRITONSERVER_TYPE_UINT8) {
        return DT_UINT8;
    } else if (data_type == TRITONSERVER_TYPE_UINT16) {
        return DT_UINT16;
    } else if (data_type == TRITONSERVER_TYPE_UINT32) {
        return DT_UINT32;
    } else if (data_type == TRITONSERVER_TYPE_UINT64) {
        return DT_UINT64;
    } else if (data_type == TRITONSERVER_TYPE_INT8) {
        return DT_INT8;
    } else if (data_type == TRITONSERVER_TYPE_INT16) {
        return DT_INT16;
    } else if (data_type == TRITONSERVER_TYPE_INT32) {
        return DT_INT32;
    } else if (data_type == TRITONSERVER_TYPE_INT64) {
        return DT_INT64;
    } else if (data_type == TRITONSERVER_TYPE_FP16) {
        return DT_FP16;
    } else if (data_type == TRITONSERVER_TYPE_FP32) {
        return DT_FP32;
    } else if (data_type == TRITONSERVER_TYPE_FP64) {
        return DT_FP64;
    } else if (data_type == TRITONSERVER_TYPE_BYTES) {
        return DT_BYTES;
    } else if (data_type == TRITONSERVER_TYPE_BF16) {
        return DT_BF16;
    } else {
        return DT_INVALID;
    }
}

TRITONSERVER_DataType to_triton_datatype(Tensor_t data_type)
{
    if (data_type == DT_INVALID) {
        return TRITONSERVER_TYPE_INVALID;
    } else if (data_type == DT_BOOL) {
        return TRITONSERVER_TYPE_BOOL;
    } else if (data_type == DT_UINT8) {
        return TRITONSERVER_TYPE_UINT8;
    } else if (data_type == DT_UINT16) {
        return TRITONSERVER_TYPE_UINT16;
    } else if (data_type == DT_UINT32) {
        return TRITONSERVER_TYPE_UINT32;
    } else if (data_type == DT_UINT64) {
        return TRITONSERVER_TYPE_UINT64;
    } else if (data_type == DT_INT8) {
        return TRITONSERVER_TYPE_INT8;
    } else if (data_type == DT_INT16) {
        return TRITONSERVER_TYPE_INT16;
    } else if (data_type == DT_INT32) {
        return TRITONSERVER_TYPE_INT32;
    } else if (data_type == DT_INT64) {
        return TRITONSERVER_TYPE_INT64;
    } else if (data_type == DT_FP16) {
        return TRITONSERVER_TYPE_FP16;
    } else if (data_type == DT_FP32) {
        return TRITONSERVER_TYPE_FP32;
    } else if (data_type == DT_FP64) {
        return TRITONSERVER_TYPE_FP64;
    } else if (data_type == DT_BYTES) {
        return TRITONSERVER_TYPE_BYTES;
    } else if (data_type == DT_BF16) {
        return TRITONSERVER_TYPE_BF16;
    } else {
        return TRITONSERVER_TYPE_INVALID;
    }
}


/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  template <typename T>
  T GetParameter(const std::string& name) {
    assert(false);
  }

  virtual ~ModelState() = default;

  common::TritonJson::Value& GetModelConfig();

 private:

  common::TritonJson::Value model_config_;
  std::shared_ptr<nvinfer1::ILogger> mTrtLogger{};

  ModelState(TRITONBACKEND_Model* triton_model, TritonJson::Value&& model_config) : BackendModel(triton_model, true), model_config_(std::move(model_config))
  {
    mTrtLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
    initLibNvInferPlugins(mTrtLogger.get(), "tensorrt_llm");
  }
};


TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));


  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);


  try {
    *state = new ModelState(triton_model, std::move(model_config));
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

common::TritonJson::Value&
ModelState::GetModelConfig() { return model_config_; }

template <>
std::string
ModelState::GetParameter<std::string>(
    const std::string& name)
{
  //TODO: Error handling
  TritonJson::Value parameters;
  model_config_.MemberAsObject("parameters", &parameters);
  TritonJson::Value value;
  std::string str_value;
  parameters.MemberAsObject(name.c_str(), &value);
  value.MemberAsString("string_value", &str_value);
  return str_value;
}

template <>
int32_t
ModelState::GetParameter<int32_t>(const std::string& name)
{
  return std::stoi(GetParameter<std::string>(name));
}

template <>
uint32_t
ModelState::GetParameter<uint32_t>(const std::string& name)
{
  return (uint32_t)std::stoul(GetParameter<std::string>( name));
}

template <>
int64_t
ModelState::GetParameter<int64_t>(const std::string& name)
{
  return std::stoll(GetParameter<std::string>(name));
}

template <>
uint64_t
ModelState::GetParameter<uint64_t>(const std::string& name)
{
  return std::stoull(GetParameter<std::string>(name));
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
  public:
    WorkItem(TRITONBACKEND_Request* request, bool isDecoupled)
    {
      mRequestId = (rand() % INT64_MAX) + 1;
      mInferenceRequest = createInferenceRequest(request, mRequestId, isDecoupled);

      // Create response factory for this request
      TRITONBACKEND_ResponseFactoryNew(&factory_ptr_, request);
    }
    WorkItem(std::shared_ptr<InferenceRequest> ir,
        uint64_t RequestId)
        : mInferenceRequest(ir),
          mRequestId(RequestId)
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

 private:

    // Convert info from original backend request to data structures defined in common/common.h
    std::shared_ptr<InferenceRequest> createInferenceRequest(TRITONBACKEND_Request* request, uint64_t requestId, bool isDecoupled)
    {
      auto inferenceRequest = std::make_shared<InferenceRequest>(requestId);

      // Extract input tensors
      std::map<std::string, tensorrt_llm::batch_manager::Tensor> input_tensors;
      uint32_t num_inputs;
      LOG_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &num_inputs), "Error getting input count");
      for (uint32_t idx = 0;  idx < num_inputs;  ++idx)
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

        //TODO: Should those be ignored?
        if (std::string(input_name) == "START" || std::string(input_name) == "CORRID" || std::string(input_name) == "END") {
            continue;
        }

        std::vector<int64_t> shapev;
        for (uint32_t i = 0;  i < dims_count;  ++i) {
          shapev.push_back(shape[i]);
        }

        auto t = tensorrt_llm::batch_manager::Tensor(input_name, MT_HOST, to_common_datatype(data_type), shapev);
        int64_t buffer_offset = 0;
        for (int64_t buffer_id=0; buffer_id < buffer_count; ++buffer_id)
        {
          const void* buffer = 0L;
          uint64_t buffer_byte_size = 0;
          TRITONSERVER_MemoryType memory_type;
          int64_t memory_type_id = 0;
          TRITONBACKEND_InputBuffer(input, buffer_id, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);
          assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));
          // TODO: Do we need to handle GPU mem input buffers??
          t.raw_copy_from(buffer, buffer_byte_size, buffer_offset);
          buffer_offset += buffer_byte_size;
        }

        inferenceRequest->emplaceInputTensor(std::string(input_name), std::move(t));
      }

      // Streaming is disabled by default. It can be enabled if request includes parameter "Streaming"
      // and this parameter is either non-zero int or true bool.
      bool streaming_flag = false;
      uint32_t num_params = 0;
      TRITONBACKEND_RequestParameterCount(request, &num_params);
      for (uint32_t param_index = 0;  param_index < num_params;  ++param_index)
      {
        const char* key = 0L;
        TRITONSERVER_ParameterType param_type;
        const void* vvalue = 0L;
        TRITONBACKEND_RequestParameter(request, param_index, &key, &param_type, &vvalue);
        if (std::string(key) == "Streaming") {
          if ((param_type == TRITONSERVER_PARAMETER_BOOL && (int)(*((char*)vvalue)) != 0) ||
              (param_type == TRITONSERVER_PARAMETER_INT && *((int*)vvalue) != 0))
          {
            streaming_flag = true;
          }
        }
      }

      inferenceRequest->setIsStreaming(streaming_flag);

      if (streaming_flag && !isDecoupled) {
          throw std::runtime_error("Streaming is only supported if model is deployed using decoupled mode.");
      }

      return inferenceRequest;
    }

    std::shared_ptr<InferenceRequest> mInferenceRequest;
    TRITONBACKEND_ResponseFactory* factory_ptr_;
    uint64_t mRequestId;
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState()
  {
    // terminate decoupled execution loop
    {
      std::lock_guard<std::mutex> lk(mWorkItemsMutex);
      mWorkItems.clear();
    }
  }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  bool isDecoupled() const
  {
      return mIsDecoupled;
  }

  bool enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count, bool isDecoupled)
  {
    try {
        std::lock_guard<std::mutex> lk(mWorkItemsMutex);
        for (uint32_t r = 0; r < request_count; ++r) {
          TRITONBACKEND_Request* request = requests[r];
          mWorkItems.emplace_back(std::make_shared<WorkItem>(request, isDecoupled));
        }
    } catch (const std::exception& e) {
        TLLM_LOG_ERROR("Error creating work item");
        return true;
    }
    return false; // return true if any error occured
  }

  // Return up to max_num_requests inference requests.
  std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(const int max_num_requests)
  {
    std::list<std::shared_ptr<InferenceRequest>> rval;
    if (max_num_requests > 0)
    {
      auto world_size = getCommWorldSize();
      auto rank = getCommWorldRank();
      if (rank == 0)
      {
        std::lock_guard<std::mutex> lk(mWorkItemsMutex);
        if (world_size > 1)
        {
          int64_t rval_size = static_cast<int64_t>(mWorkItems.size());
          bcast(&rval_size, 1, MPI_TYPE_INT64_T, 0);
        }
        if (mWorkItems.size() > 0)
        {
          int count = 0;
          while (count < max_num_requests && mWorkItems.size() > 0)
          {
            auto work_item = mWorkItems.front();
            mWorkItems.pop_front();
            rval.emplace_back(work_item->getInferenceRequest());
            mWorkItemsInProgress.emplace(std::make_pair(work_item->requestId(), work_item));
            count++;
          }
          if (world_size > 1)
          {
            std::vector<int64_t> packed;
            for (auto ir : rval)
            {
              auto vpacked = ir->serialize();
              packed.push_back(static_cast<int64_t>(vpacked.size()));
              packed.insert(packed.end(),
              std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
            }
            int64_t nWords1 = static_cast<int64_t>(packed.size());
            bcast(&nWords1, 1, MPI_TYPE_INT64_T, 0);
                  bcast(packed, 0);
          }
        }
      }
      else
      {
        // subordinate ranks hang until master rank sends work
        int64_t rval_size;
        bcast(&rval_size, 1, MPI_TYPE_INT64_T, 0);
        if (rval_size > 0)
        {
          int nWords1;
          bcast(&nWords1, 1, MPI_TYPE_INT64_T, 0);
          std::vector<int64_t> packed(nWords1);
          bcast(packed, 0);
          int64_t* packed_ptr = packed.data();
          for (int64_t count = 0; count < rval_size; ++count)
          {
            int64_t n = *(packed_ptr++);
            auto ir = InferenceRequest::deserialize(packed_ptr);
            packed_ptr += n;
            rval.emplace_back(ir);
          }
        }
      }
    }
    return rval;
  }

  TRITONSERVER_Error* sendTritonResponse(uint64_t requestId, std::list<std::shared_ptr<tensorrt_llm::batch_manager::Tensor>> const& response_tensors, bool final_response, const std::string& errMsg)
  {
    if (getCommWorldRank() == 0)
    {
        TRITONBACKEND_ResponseFactory* response_factory;
        {
            std::lock_guard<std::mutex> lk(mWorkItemsMutex);
            auto work_item = mWorkItemsInProgress.at(requestId);
            response_factory = work_item->response_factory();
        }

        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));

        if (final_response)
        {
            std::lock_guard<std::mutex> lk(mWorkItemsMutex);
            mWorkItemsInProgress.erase(requestId);
        }

        // Check if error
        TRITONSERVER_Error* err = nullptr;
        if (!errMsg.empty()) {
            std::string errStr = "Encountered error for requestId " + std::to_string(requestId) + ": " + errMsg;
            TLLM_LOG_ERROR(errStr);

            err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
            final_response = true;
        } else {
            for (auto it = response_tensors.begin();  it != response_tensors.end();  ++it)
            {
              auto tensor = *it;
              auto shape = tensor->shape(); // returns std::vectorint64_t>
              TRITONBACKEND_Output* output;
              RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
                      response, &output, tensor->name().c_str(), to_triton_datatype(tensor->datatype()),
                      shape.data(), shape.size()));

              uint64_t buffersize = tensor->sizeBytes();
              void* buffer = 0L;
              TRITONSERVER_MemoryType memory_type;
              int64_t memory_type_id;
              RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, buffersize, &memory_type, &memory_type_id));
              tensor->raw_copy_to(buffer, buffersize, 0L);
            }
        }

        RETURN_IF_ERROR(
            TRITONBACKEND_ResponseSend(
              response, final_response ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, err));
      }


      return nullptr;
  }

  void sendResponse(uint64_t requestId, std::list<std::shared_ptr<tensorrt_llm::batch_manager::Tensor>> const& response_tensors, bool final_response, const std::string& errMsg)
  {
    auto tritonErr = sendTritonResponse(requestId, response_tensors, final_response, errMsg);
    LOG_IF_ERROR(tritonErr, "Failed to send Triton response for requestId: " + std::to_string(requestId));
    return;
  }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state), mIsDecoupled(false)
  {
      // Note: std::string::compare fails this test (always return non-zero value).
      // Using old school strcmp instead.
      if (model_state_->GetParameter<std::string>("gpt_model_type") == "V1" ||
          model_state_->GetParameter<std::string>("gpt_model_type") == "v1")
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
        throw std::runtime_error("Invalid gpt_model_type. Must be v1/inflight_batching/inflight_fused_batching.");
      }

      // Check if model is in decoupled mode:
      triton::common::TritonJson::Value transaction_policy;
      model_state_->GetModelConfig().MemberAsObject("model_transaction_policy", &transaction_policy);
      transaction_policy.MemberAsBool("decoupled", &mIsDecoupled);

      // Note: std::string::compare fails this test (always return non-zero value).
      // Using old school strcmp instead.
      mModelPath = model_state_->GetParameter<std::string>("gpt_model_path");
      auto configPath = mModelPath + "/config.json";
      std::ifstream jsonStream(configPath);

      auto constexpr allowExceptions = true;
      auto constexpr ingoreComments = true;
      auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);

      auto const& builderConfig = json.at("builder_config");
      int maxInputLen = builderConfig.at("max_input_len");
      int maxOutputLen = builderConfig.at("max_output_len");
      int maxSeqLen = maxInputLen + maxOutputLen;
      int maxNumRequests = builderConfig.at("max_batch_size");
      int32_t maxBeamWidth = model_state_->GetParameter<int32_t>("max_beam_width");

      mBatchManager = std::make_shared<GptManager>(mModelPath, mTrtGptModelType, maxSeqLen, maxNumRequests, maxBeamWidth,
          [this](int max_num_requests){return get_inference_requests(max_num_requests);},
          [this](uint64_t requestId, std::list<std::shared_ptr<tensorrt_llm::batch_manager::Tensor>> response_tensors, bool final_response, const std::string& errMsg){return sendResponse(requestId, response_tensors, final_response, errMsg);});

      if (getCommWorldRank() != 0)
      {
          while (true) {}
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

  // Initialize to clearly invalid values, will be overwritten later.
  std::shared_ptr<GptManager> mBatchManager;

  std::list<std::shared_ptr<WorkItem>> mWorkItems;
  std::unordered_map<uint64_t, std::shared_ptr<WorkItem>> mWorkItemsInProgress;
  std::mutex mWorkItemsMutex;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
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
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  auto isDecoupled = instance_state->isDecoupled();

  RETURN_ERROR_IF_TRUE(
      instance_state->enqueue(requests, request_count, isDecoupled),
      TRITONSERVER_ERROR_INTERNAL,
      std::string("unexpected error in enqueue method"));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
  }

  return nullptr;  // success
}

}  // extern "C"


}}}  // namespace triton::backend::minimal
