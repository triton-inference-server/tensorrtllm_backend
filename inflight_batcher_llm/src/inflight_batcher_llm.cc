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

using namespace ::triton::common;   // TritonJson

//
// Mockup of LLM inflight batcher based on triton 'minimal' backend example
//

using namespace inflight_batcher::common;
using namespace inflight_batcher::batch_manager;
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

//
// Minimal backend that demonstrates the TRITONBACKEND API. This
// backend works for any model that has 1 input called "IN0" with
// INT32 datatype and shape [ 4 ] and 1 output called "OUT0" with
// INT32 datatype and shape [ 4 ]. The backend supports both batching
// and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

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

  TRITONSERVER_Error* ValidateModelConfig();

  template <typename T>
  T GetParameter(const std::string& name) {
    assert(false);
  }

  virtual ~ModelState() = default;

 private:

  common::TritonJson::Value model_config_;
  std::shared_ptr<nvinfer1::ILogger> mTrtLogger{};

  ModelState(TRITONBACKEND_Model* triton_model, TritonJson::Value&& model_config) : BackendModel(triton_model), model_config_(std::move(model_config))
  {
    mTrtLogger = std::make_shared<TllmLogger>();
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
    WorkItem(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request* request)
    {
      instance_ptr_ = instance;
      // TODO: Is there need to explicitly take over responsibility for request?
      request_ptr_ = request;
      TRITONBACKEND_RequestCorrelationId(request, &correlation_id_);
      // Create response factory for this request
      TRITONBACKEND_ResponseFactoryNew(&factory_ptr_, request);
    }
    ~WorkItem()
    {
      // TODO: Verify if this is right way to release request.
      TRITONBACKEND_RequestRelease(request_ptr_, TRITONSERVER_REQUEST_RELEASE_ALL);
      TRITONBACKEND_ResponseFactoryDelete(factory_ptr_);
    }

    TRITONBACKEND_ModelInstance* instance()
    {
      return instance_ptr_;
    }

    TRITONBACKEND_Request* request()
    {
      return request_ptr_;
    }

    TRITONBACKEND_ResponseFactory* response_factory()
    {
      return factory_ptr_;
    }

    uint64_t correlation_id()
    {
      return correlation_id_;
    }

  private:
    TRITONBACKEND_ModelInstance* instance_ptr_;
    TRITONBACKEND_Request* request_ptr_;
    TRITONBACKEND_ResponseFactory* factory_ptr_;
    uint64_t correlation_id_;
    // TODO: Add private state object. Not needed for this experiment.
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
      std::lock_guard<std::mutex> lk(work_items_m_);
      work_items_.clear();
    }
  }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Convert info from original backend request to data structures defined in common/common.h
  std::shared_ptr<InferenceRequest> convert(std::shared_ptr<WorkItem> work_item)
  {
    // Extract input tensors
    std::map<std::string, Tensor> input_tensors;
    TRITONBACKEND_Request* request = work_item->request();
    uint32_t num_inputs;
    TRITONBACKEND_RequestInputCount(request, &num_inputs);
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

      auto t = Tensor(input_name, MT_HOST, to_common_datatype(data_type), shapev);
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

      input_tensors.emplace(std::make_pair(std::string(input_name), t));
    }

    // Get correlation id
    uint64_t correlation_id;
    TRITONBACKEND_RequestCorrelationId(request, &correlation_id);
    return std::make_shared<InferenceRequest>(input_tensors, correlation_id);
  }

  bool enqueue(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
  {
    std::lock_guard<std::mutex> lk(work_items_m_);
    for (uint32_t r = 0; r < request_count; ++r) {
      TRITONBACKEND_Request* request = requests[r];
      work_items_.emplace_back(std::make_shared<WorkItem>(instance, request));
      uint32_t num_params = 0;
      TRITONBACKEND_RequestParameterCount(request, &num_params);
      if (num_params > 0)
      {
        for (uint32_t idx = 0;  idx < num_params;  ++idx)
        {
          const char* key = 0L;
          TRITONSERVER_ParameterType type;
          const void* vvalue = 0L;
          TRITONBACKEND_RequestParameter(request, idx, &key, &type, &vvalue);
          switch (type)
          {
            case TRITONSERVER_PARAMETER_STRING:
              //printf("%d :: --> \"%s\": %s\n",__LINE__,key,(const char*)vvalue);
              break;
            case TRITONSERVER_PARAMETER_INT:
              //printf("%d :: --> \"%s\": %d\n",__LINE__,key,*((int*)vvalue));
              break;
            case TRITONSERVER_PARAMETER_BOOL:
              //printf("%d :: --> \"%s\": %d\n",__LINE__,key,(int)(*((char*)vvalue)));
              break;
            case TRITONSERVER_PARAMETER_BYTES:
              //printf("%d :: --> \"%s\": %d\n",__LINE__,key,(int)(*((char*)vvalue)));
              break;
            default:
              assert(false);
              break;
          }
        }
      }
    }
    return false; // return true if any error occured
  }

  // Return up to max_num_requests inference requests.
  std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(int max_num_requests)
  {
    std::lock_guard<std::mutex> lk(work_items_m_);
    std::list<std::shared_ptr<InferenceRequest>> rval;
    if (max_num_requests <= 0) max_num_requests = (int)work_items_.size();

    int count = 0;
    while (count < max_num_requests && work_items_.size() > 0)
    {
      auto work_item = work_items_.front();
      work_items_.pop_front();
      rval.emplace_back(convert(work_item));
      work_items_in_progress_.emplace(std::make_pair(work_item->correlation_id(), work_item));

      ++count;
    }
    if (rval.size() > 0)
      std::cout << "Returning " << rval.size() << " inference requests" << std::endl;

    return rval;
  }

  // Move work item from work_items_in_progress_ to work_items_.
  // This is done by GPT manager if it isn't able to process it (most likely due to some error),
  // but want to try again later.
  void restore_inference_request(std::vector<uint64_t> correlation_ids)
  {
    // No-op, will probably be removed from API
  }

  // Remove work from work_items_in_progress_ map. This will free WorkItem.
  // This method should be called only after final response has been sent.
  void commit_inference_request(std::vector<uint64_t> correlation_ids)
  {
    // No-op, will probably be removed from API
  }

  TRITONSERVER_Error* send_response_(uint64_t correlation_id, std::list<std::shared_ptr<Tensor>> response_tensors, bool final_response)
  {
    auto work_item = work_items_in_progress_[correlation_id];
    auto response_factory = work_item->response_factory();
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));

    for (auto it = response_tensors.begin();  it != response_tensors.end();  ++it)
    {
      auto tensor = *it;
      auto shape = tensor->shape(); // returns std::vectorint64_t>
      TRITONBACKEND_Output* output;
      TRITONBACKEND_ResponseOutput(
              response, &output, tensor->name().c_str(), to_triton_datatype(tensor->datatype()),
              shape.data(), shape.size());

      uint64_t buffersize = tensor->sizeBytes();
      void* buffer = 0L;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      TRITONBACKEND_OutputBuffer(output, &buffer, buffersize, &memory_type, &memory_type_id);
      tensor->raw_copy_to(buffer, buffersize, 0L);
    }

    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
          response, final_response ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, nullptr),
        "failed to send response");
    if (final_response)
    {
      work_items_in_progress_.erase(correlation_id);
    }
    return 0L;
  }

  bool send_response(uint64_t correlation_id, std::list<std::shared_ptr<Tensor>> response_tensors, bool final_response)
  {
    if (send_response_(correlation_id, response_tensors, final_response))
    {
      return false;
    } else {
      return true;
    }
  }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
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
      else
      {
        throw std::runtime_error("Invalid gpt_model_type. Must be v1 or inflight_batching.");
      }

      // Note: std::string::compare fails this test (always return non-zero value).
      // Using old school strcmp instead.
      mModelPath = model_state_->GetParameter<std::string>("gpt_model_path");

      mBatchManager = std::make_shared<GptManager>(mModelPath, mTrtGptModelType, mMaxSeqLen, mMaxNumRequests,
          [this](int max_num_requests){return get_inference_requests(max_num_requests);},
          [this](std::vector<uint64_t> correlation_ids){commit_inference_request(correlation_ids);},
          [this](std::vector<uint64_t> correlation_ids){restore_inference_request(correlation_ids);},
          [this](uint64_t correlation_id, std::list<std::shared_ptr<Tensor>> response_tensors, bool final_response){return send_response(correlation_id, response_tensors, final_response);});
  }

  ModelState* model_state_;

  //
  // inflight batcher is a decoupled design.
  // It uses response factory objects to decouple responses from requests.
  //
  // New requests are added to work_items_ list. This list is processed
  // in an infinite loop run by a worker thread. Requests take multiple
  // iterations to complete, and number of iterations is not known in
  // advance. To facilitate this, we use response factory objects to
  // decouple requests and responses.
  //
  TrtGptModelType mTrtGptModelType;
  std::string mModelPath;

  // TODO: Those should come from config
  int mMaxSeqLen = 60;
  int mMaxNumRequests = 32;
  std::shared_ptr<GptManager> mBatchManager;

  std::list<std::shared_ptr<WorkItem>> work_items_;
  std::unordered_map<uint64_t, std::shared_ptr<WorkItem>> work_items_in_progress_;
  std::mutex work_items_m_;
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
  RETURN_ERROR_IF_TRUE(
      instance_state->enqueue(instance, requests, request_count),
      TRITONSERVER_ERROR_INTERNAL,
      std::string("unexpected error in enqueue method"));
  return nullptr;  // success
}

}  // extern "C"


}}}  // namespace triton::backend::minimal
