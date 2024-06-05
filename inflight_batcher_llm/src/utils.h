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

#include "NvInfer.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <map>
#include <string>
#include <unordered_set>

using namespace tensorrt_llm;

namespace triton::backend::inflight_batcher_llm
{

/// @brief Names of input fields
struct InputFieldsNames
{
    static constexpr char const* inputTokens = "input_ids";
    static constexpr char const* maxNewTokens = "request_output_len";
    static constexpr char const* endId = "end_id";
    static constexpr char const* padId = "pad_id";
    static constexpr char const* badWords = "bad_words_list";
    static constexpr char const* stopWords = "stop_words_list";
    static constexpr char const* embeddingBias = "embedding_bias";

    // OutputConfig
    static constexpr char const* returnLogProbs = "return_log_probs";
    static constexpr char const* returnGenerationLogits = "return_generation_logits";
    static constexpr char const* returnContextLogits = "return_context_logits";

    // SamplingConfig
    static constexpr char const* beamWidth = "beam_width";
    static constexpr char const* topK = "runtime_top_k";
    static constexpr char const* topP = "runtime_top_p";
    static constexpr char const* topPMin = "runtime_top_k_min";
    static constexpr char const* topPDecay = "runtime_top_p_decay";
    static constexpr char const* topPResetIds = "runtime_top_p_reset_ids";
    static constexpr char const* temperature = "temperature";
    static constexpr char const* lengthPenalty = "len_penalty";
    static constexpr char const* earlyStopping = "early_stopping";
    static constexpr char const* repetitionPenalty = "repetition_penalty";
    static constexpr char const* minLength = "min_length";
    static constexpr char const* beamSearchDiversityRate = "beam_search_diversity_rate";
    static constexpr char const* presencePenalty = "presence_penalty";
    static constexpr char const* frequencyPenalty = "frequency_penalty";
    static constexpr char const* randomSeed = "random_seed";

    // PromptTuningConfig
    static constexpr char const* promptEmbeddingTable = "prompt_embedding_table";

    // LoraConfig
    static constexpr char const* loraTaskId = "lora_task_id";
    static constexpr char const* loraWeights = "lora_weights";
    static constexpr char const* loraConfig = "lora_config";

    // SpeculativeDecodingConfig
    static constexpr char const* draftInputs = "draft_input_ids";
    static constexpr char const* draftLogits = "draft_logits";
    static constexpr char const* draftAcceptanceThreshold = "draft_acceptance_threshold";
};

/// @brief Names of output fields
struct OutputFieldsNames
{
    static constexpr char const* outputIds = "output_ids";
    static constexpr char const* sequenceLength = "sequence_length";
    static constexpr char const* contextLogits = "context_logits";
    static constexpr char const* generationLogits = "generation_logits";
    static constexpr char const* outputLogProbs = "output_log_probs";
    static constexpr char const* cumLogProbs = "cum_log_probs";
};

inline static std::string const kStopInputTensorName = "stop";
inline static std::string const kStreamingInputTensorName = "streaming";

namespace utils
{

/// @brief  Convert Triton datatype to TRT datatype
nvinfer1::DataType to_trt_datatype(TRITONSERVER_DataType data_type);

using InputTensors = std::unordered_map<std::string, tensorrt_llm::batch_manager::NamedTensor>;

/// @brief Gather input tenors in a Triton request
/// @return An unordered map with key being input name and value being input tensor
InputTensors readInputsTensors(TRITONBACKEND_Request* request);

/// @brief Construct executor::SampleConfig from input tensors
executor::SamplingConfig getSamplingConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::OutputConfig from input tensors
executor::OutputConfig getOutputConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::SpeculativeDecodingConfig from input tensors
std::optional<executor::SpeculativeDecodingConfig> getSpeculativeDecodingConfigFromTensors(
    InputTensors const& inputsTensors);

/// @brief Construct executor::PromptTuningConfig from input tensors
std::optional<executor::PromptTuningConfig> getPromptTuningConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::LoraConfig from input tensors
std::optional<executor::LoraConfig> getLoraConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::Request from input tensors
executor::Request createRequestFromInputTensors(
    std::unordered_map<std::string, tensorrt_llm::batch_manager::NamedTensor> const& inputsTensors,
    bool excludeInputFromOutput, bool isDecoupled, bool streaming);

/// @brief get the requestId of the request and update requestIdStrMap
/// @return Returns 0 if not specified. Throws an error if request_id cannot be convert to uint64_t
uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap);

/// @brief Get the requested output names
std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request);

/// @brief Get the value of a boolean tensor
bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, std::string const& inputTensorName);

/// @brief Get a single value tensor from the input tensors
/// @return true if the value is found else false
template <typename Value>
bool extractSingleton(std::unordered_map<std::string, tensorrt_llm::batch_manager::NamedTensor> const& params,
    std::string const& name, Value& value)
{
    if (!params.count(name))
    {
        return false;
    }
    auto const& tensor = params.at(name);
    TLLM_CHECK_WITH_INFO(tensor.tensor->getSize() == 1, "Invalid size for tensor " + name);
    value = *(static_cast<Value*>(tensor.tensor->data()));
    return true;
}

/// @brief Get a single value tensor from the input tensors and put it into an optional. Set to std::nullopt if it's not
/// found.
template <typename Value>
void extractOptionalSingleton(std::unordered_map<std::string, tensorrt_llm::batch_manager::NamedTensor> const& params,
    std::string const& name, std::optional<Value>& optionalValue)
{
    Value value;
    if (extractSingleton<Value>(params, name, value))
    {
        optionalValue = value;
    }
    else
    {
        optionalValue = std::nullopt;
    }
}

/// @brief Get a 1d tensor from the input tensors
/// @return true if the tensor is found else false
template <typename Value>
bool extractVector(std::unordered_map<std::string, tensorrt_llm::batch_manager::NamedTensor> const& params,
    std::string const& name, std::vector<Value>& value)
{
    if (!params.count(name))
    {
        return false;
    }
    auto const& tensor = params.at(name);
    int64_t n = tensor.tensor->getSize();
    value.resize(n);
    for (int64_t i = 0; i < n; ++i)
    {
        value[i] = static_cast<Value*>(tensor.tensor->data())[i];
    }
    return true;
}

int64_t numElements(std::vector<int64_t> const& shape);

/// @brief Flatten the vector and copy into the buffer
template <typename T>
void flatten(std::vector<T> const& vec, void* buffer, std::vector<int64_t> const& expectedShape)
{
    TLLM_CHECK_WITH_INFO(static_cast<int64_t>(vec.size()) == numElements(expectedShape),
        "Trying to flatten a tensor with unexpected size");
    T* typedBuffer = static_cast<T*>(buffer);
    std::copy(vec.begin(), vec.end(), typedBuffer);
}

/// @brief Flatten the vector of vector and copy into the buffer
template <typename T>
void flatten(std::vector<std::vector<T>> const& vec, void* buffer, std::vector<int64_t> const& expectedShape)
{
    T* typedBuffer = static_cast<T*>(buffer);
    int64_t copiedSize = 0;
    for (auto const& innerVec : vec)
    {
        TLLM_CHECK_WITH_INFO(innerVec.size() == vec.at(0).size(),
            "The vector of vector to be flattened has mismatched sizes in its inner vectors");
        copiedSize += innerVec.size();
        typedBuffer = std::copy(innerVec.begin(), innerVec.end(), typedBuffer);
    }
    TLLM_CHECK_WITH_INFO(copiedSize == numElements(expectedShape), "Trying to flatten a tensor with unexpected size");
}

/// @brief Flatten the tensor and copy into the buffer
template <typename Value>
void flatten(tensorrt_llm::executor::Tensor const& tensor, void* buffer, std::vector<int64_t> const& expectedShape)
{
    TLLM_CHECK_WITH_INFO(static_cast<int64_t>(tensor.getSize()) == numElements(expectedShape),
        "Trying to flatten a tensor with unexpected size");
    Value* typedBuffer = static_cast<Value*>(buffer);
    Value const* ptr = static_cast<Value const*>(tensor.getData());
    std::copy(ptr, ptr + tensor.getSize(), typedBuffer);
}

/// @brief Query Triton for a buffer that can be used to pass the output tensors
template <typename T>
void* getResponseBuffer(TRITONBACKEND_Response* tritonResponse, std::vector<int64_t> const& shape,
    TRITONSERVER_DataType dtype, std::string const& name)
{
    TRITONBACKEND_Output* output;
    TRITONSERVER_Error* err{nullptr};
    err = TRITONBACKEND_ResponseOutput(tritonResponse, &output, name.c_str(), dtype, shape.data(), shape.size());
    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        TLLM_THROW("Could not get response output for output tensor %s: %s", name.c_str(), errMsg);
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    uint64_t size = 1;
    for (auto s : shape)
    {
        size *= s;
    }
    auto buffersize = size * sizeof(T);
    void* tritonBuffer = 0L;
    err = TRITONBACKEND_OutputBuffer(output, &tritonBuffer, buffersize, &memory_type, &memory_type_id);
    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        TLLM_THROW("Could not get output buffer for output tensor %s: %s", name.c_str(), errMsg);
    }
    return tritonBuffer;
}

/// @brief Convert a sparse tensor to a list of VecTokens
std::list<executor::VecTokens> convertWordList(executor::VecTokens const& sparseList);

/// @brief Remove the additional size 1 dimension for tensor
void squeezeTensor(std::shared_ptr<runtime::ITensor> const& tensor, int32_t expectedNumDims);

/// Helper functions to parse a csv delimited string to a vector ints
std::vector<int32_t> csvStrToVecInt(std::string const& str);

/// Helper functions to parse a csv delimited string to a vector of vector ints
std::vector<std::vector<int32_t>> csvStrToVecVecInt(std::string const& str);

} // namespace utils
} // namespace triton::backend::inflight_batcher_llm
