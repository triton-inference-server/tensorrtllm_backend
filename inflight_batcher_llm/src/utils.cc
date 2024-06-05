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

#include "utils.h"

using namespace tensorrt_llm::batch_manager;

namespace triton::backend::inflight_batcher_llm::utils
{

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

std::unordered_map<std::string, NamedTensor> readInputsTensors(TRITONBACKEND_Request* request)
{
    std::unordered_map<std::string, NamedTensor> inputsTensors;
    uint32_t num_inputs;
    LOG_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &num_inputs), "Error getting input count");
    for (uint32_t idx = 0; idx < num_inputs; ++idx)
    {
        TRITONBACKEND_Input* input = nullptr;
        LOG_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, idx, &input), "Error getting input index");

        char const* input_name = nullptr;
        TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_INVALID;
        int64_t const* shape = nullptr;
        uint32_t dims_count = 0;
        uint64_t byte_size = 0;
        uint32_t buffer_count = 0;
        LOG_IF_ERROR(TRITONBACKEND_InputProperties(
                         input, &input_name, &data_type, &shape, &dims_count, &byte_size, &buffer_count),
            "Error getting input properties");

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

        NamedTensor t(utils::to_trt_datatype(data_type), shapev, input_name);
        uint64_t buffer_offset = 0;
        for (int64_t buffer_id = 0; buffer_id < buffer_count; ++buffer_id)
        {
            void const* buffer = nullptr;
            uint64_t buffer_byte_size = 0;
            TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            LOG_IF_ERROR(
                TRITONBACKEND_InputBuffer(input, buffer_id, &buffer, &buffer_byte_size, &memory_type, &memory_type_id),
                "failed to get input buffer");
            assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));
            std::memcpy(static_cast<char*>(t.tensor->data()) + buffer_offset, buffer, buffer_byte_size);
            buffer_offset += buffer_byte_size;
        }

        inputsTensors.insert(make_pair(t.name, std::move(t)));
    }
    return inputsTensors;
}

uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap)
{
    char const* charRequestId;
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
            catch (std::exception const& e)
            {
                std::hash<std::string> hasher;
                requestId = hasher(strRequestId);

                // Check for hash collisions
                // If requestID already exists in the map with the same string, increment the ID and check again
                for (auto it = requestIdStrMap.find(requestId);
                     it != requestIdStrMap.end() && it->second != strRequestId;)
                {
                    requestId++;
                }
            }
            requestIdStrMap.insert({requestId, strRequestId});
        }
    }

    return requestId;
}

std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request)
{
    std::unordered_set<std::string> outputNames;
    uint32_t outputCount;
    LOG_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &outputCount), "Error getting request output count");
    for (size_t i = 0; i < outputCount; ++i)
    {
        char const* name;
        LOG_IF_ERROR(TRITONBACKEND_RequestOutputName(request, i, &name), "Error getting request output name");
        std::string name_s(name);
        outputNames.insert(std::move(name_s));
    }
    return outputNames;
}

bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, std::string const& inputTensorName)
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
        TRITONSERVER_ErrorDelete(error);
        return false;
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
        ("ModelInstanceState::getRequestStopSignal: buffer_count = " + std::to_string(buffer_count)).c_str());

    void const* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));

    bool boolean = *reinterpret_cast<bool const*>(buffer);

    return boolean;
}

std::string sparseListToStr(executor::VecTokens const& sparseList)
{
    std::string buffer;
    for (auto v : sparseList)
    {
        buffer.append(std::to_string(v) + " ");
    }
    return buffer;
}

std::list<executor::VecTokens> convertWordList(executor::VecTokens const& sparseList)
{
    std::list<executor::VecTokens> convertedList;
    int32_t n = sparseList.size();
    TLLM_CHECK_WITH_INFO(n % 2 == 0, "Sparse list must not have odd length: " + sparseListToStr(sparseList));
    int32_t numTokens = n / 2;
    int32_t currentIndex = 0;
    for (auto i = numTokens; i < n; ++i)
    {
        if (sparseList[i] == -1)
        {
            for (auto j = i + 1; j < n; ++j)
            {
                TLLM_CHECK_WITH_INFO(
                    sparseList[j] == -1, "Sparse list must not have additional -1s: " + sparseListToStr(sparseList));
            }
            break;
        }
        TLLM_CHECK_WITH_INFO(sparseList[i] <= numTokens,
            "Sparse list must not have out-of-bound offsets: " + sparseListToStr(sparseList));
        if (i != numTokens)
        {
            TLLM_CHECK_WITH_INFO(sparseList[i] > sparseList[i - 1],
                "Sparse list must not have non-increasing offsets: " + sparseListToStr(sparseList));
        }
        executor::VecTokens currentWords;
        while (currentIndex < sparseList[i])
        {
            currentWords.push_back(sparseList[currentIndex]);
            ++currentIndex;
        }
        convertedList.push_back(currentWords);
    }
    return convertedList;
}

void squeezeTensor(std::shared_ptr<runtime::ITensor> const& tensor, int32_t expectedNumDims)
{
    auto shape = tensor->getShape();
    if (shape.nbDims == expectedNumDims)
    {
        return;
    }
    if (shape.nbDims == expectedNumDims + 1 && shape.d[0] == 1)
    {
        --shape.nbDims;
        for (int32_t i = 0; i < expectedNumDims; ++i)
        {
            shape.d[i] = shape.d[i + 1];
        }
        tensor->reshape(shape);
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected prompt tensor shape");
    }
}

std::vector<int32_t> csvStrToVecInt(std::string const& str)
{
    TLLM_CHECK_WITH_INFO(!str.empty(), "Cannot convert empty string to vector of vector of ints");

    std::vector<int32_t> output;
    std::stringstream ss(str);
    while (ss.good())
    {
        std::string substr;
        ss >> std::ws;
        getline(ss, substr, ',');
        if (substr.empty())
        {
            break;
        }
        output.push_back(std::stoi(substr));
    }
    TLLM_CHECK_WITH_INFO(!output.empty(), "Empty vector");
    return output;
}

std::vector<std::vector<int32_t>> csvStrToVecVecInt(std::string const& str)
{
    TLLM_CHECK_WITH_INFO(!str.empty(), "Cannot convert empty string to vector of vector of ints");

    std::vector<std::vector<int32_t>> output;
    std::stringstream ss(str);

    while (true)
    {
        std::string substr;
        getline(ss, substr, '}');
        if (substr.empty() || ss.eof())
        {
            break;
        }
        if (substr[0] == '{')
        {
            // Remove the opening bracket from the content
            substr = substr.substr(1);
        }
        output.push_back(csvStrToVecInt(substr));
        // Ignore the comma and any whitespace
        ss >> std::ws;
        ss.ignore(std::numeric_limits<std::streamsize>::max(), ',');
        ss >> std::ws;
    }
    TLLM_CHECK_WITH_INFO(!output.empty(), "Empty vector of vector");
    return output;
}

int64_t numElements(std::vector<int64_t> const& shape)
{
    int64_t n = 1;
    for (auto d : shape)
    {
        n *= d;
    }
    return n;
}

executor::SamplingConfig getSamplingConfigFromTensors(InputTensors const& inputsTensors)
{
    int32_t beamWidth = 1;
    // If beam_width is specified, set it from config.pbtxt
    extractSingleton<int32_t>(inputsTensors, InputFieldsNames::beamWidth, beamWidth);

    std::optional<executor::SizeType32> topK{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::topK, topK);

    std::optional<float> topP{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topP, topP);
    if (topP.has_value() && topP.value() <= 0.F)
    {
        topP.reset();
    }

    std::optional<float> topPMin{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topPMin, topPMin);

    std::optional<float> topPDecay{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topPDecay, topPDecay);

    std::optional<int32_t> topPResetIds{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::topPResetIds, topPResetIds);

    std::optional<float> temperature{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::temperature, temperature);

    std::optional<float> lengthPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::lengthPenalty, lengthPenalty);

    std::optional<int32_t> earlyStopping{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::earlyStopping, earlyStopping);

    std::optional<float> repetitionPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::repetitionPenalty, repetitionPenalty);

    std::optional<int32_t> minLength{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::minLength, minLength);

    std::optional<float> beamSearchDiversityRate{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::beamSearchDiversityRate, beamSearchDiversityRate);

    std::optional<float> presencePenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::presencePenalty, presencePenalty);

    std::optional<float> frequencyPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::frequencyPenalty, frequencyPenalty);

    std::optional<uint64_t> randomSeed{std::nullopt};
    extractOptionalSingleton<uint64_t>(inputsTensors, InputFieldsNames::randomSeed, randomSeed);

    return executor::SamplingConfig(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature,
        minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, lengthPenalty,
        earlyStopping);
}

executor::OutputConfig getOutputConfigFromTensors(InputTensors const& inputsTensors)
{
    bool returnLogProbs{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnLogProbs, returnLogProbs);

    bool returnGenerationLogits{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnGenerationLogits, returnGenerationLogits);

    bool returnContextLogits{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnContextLogits, returnContextLogits);

    // Note that currently excludeInputFromOutput is set from the backend parameters.
    return executor::OutputConfig(returnLogProbs, returnContextLogits, returnGenerationLogits);
}

std::optional<executor::SpeculativeDecodingConfig> getSpeculativeDecodingConfigFromTensors(
    InputTensors const& inputsTensors)
{
    std::optional<executor::SpeculativeDecodingConfig> speculativeDecodingConfig = std::nullopt;

    if (inputsTensors.count(InputFieldsNames::draftInputs))
    {
        executor::VecTokens draftInputs;
        extractVector<int32_t>(inputsTensors, InputFieldsNames::draftInputs, draftInputs);

        std::optional<executor::Tensor> draftLogits = std::nullopt;
        if (inputsTensors.count(InputFieldsNames::draftLogits))
        {
            std::shared_ptr<runtime::ITensor> originaldraftLogitsTensor
                = inputsTensors.at(InputFieldsNames::draftLogits).tensor;
            utils::squeezeTensor(originaldraftLogitsTensor, 2);
            draftLogits = executor::detail::ofITensor(originaldraftLogitsTensor);
        }

        std::optional<float> draftAcceptanceThreshold{std::nullopt};
        utils::extractOptionalSingleton<float>(
            inputsTensors, InputFieldsNames::draftAcceptanceThreshold, draftAcceptanceThreshold);

        speculativeDecodingConfig
            = executor::SpeculativeDecodingConfig(draftInputs, draftLogits, draftAcceptanceThreshold);
    }
    return speculativeDecodingConfig;
}

std::optional<executor::PromptTuningConfig> getPromptTuningConfigFromTensors(InputTensors const& inputsTensors)
{
    std::optional<executor::PromptTuningConfig> pTuningConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::promptEmbeddingTable))
    {
        std::shared_ptr<runtime::ITensor> originalTensor
            = inputsTensors.at(InputFieldsNames::promptEmbeddingTable).tensor;
        utils::squeezeTensor(originalTensor, 2);
        auto const& executorTensor = executor::detail::ofITensor(originalTensor);
        pTuningConfig = executor::PromptTuningConfig(executorTensor);
    }
    return pTuningConfig;
}

std::optional<executor::LoraConfig> getLoraConfigFromTensors(InputTensors const& inputsTensors)
{
    std::optional<executor::LoraConfig> loraConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::loraTaskId))
    {
        uint64_t taskId;
        if (!utils::extractSingleton<uint64_t>(inputsTensors, InputFieldsNames::loraTaskId, taskId))
        {
            throw std::runtime_error("failed to extract lora task id");
        }

        std::optional<executor::Tensor> loraConfigTensor{std::nullopt};
        if (inputsTensors.count(InputFieldsNames::loraConfig))
        {
            std::shared_ptr<runtime::ITensor> originalLoraConfigTensor
                = inputsTensors.at(InputFieldsNames::loraConfig).tensor;
            utils::squeezeTensor(originalLoraConfigTensor, 2);
            loraConfigTensor = executor::detail::ofITensor(originalLoraConfigTensor);
        }

        std::optional<executor::Tensor> loraWeightsTensor{std::nullopt};
        if (inputsTensors.count(InputFieldsNames::loraWeights))
        {
            std::shared_ptr<runtime::ITensor> originalLoraWeightsTensor
                = inputsTensors.at(InputFieldsNames::loraWeights).tensor;
            utils::squeezeTensor(originalLoraWeightsTensor, 2);
            loraWeightsTensor = executor::detail::ofITensor(originalLoraWeightsTensor);
        }

        loraConfig = executor::LoraConfig(taskId, loraWeightsTensor, loraConfigTensor);
    }
    return loraConfig;
}

executor::Request createRequestFromInputTensors(std::unordered_map<std::string, NamedTensor> const& inputsTensors,
    bool excludeInputFromOutput, bool isDecoupled, bool streaming)
{
    executor::OutputConfig outConfig = utils::getOutputConfigFromTensors(inputsTensors);
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    executor::VecTokens inputTokens;
    if (!utils::extractVector<int32_t>(inputsTensors, InputFieldsNames::inputTokens, inputTokens))
    {
        throw std::runtime_error("input_ids is not present in the request");
    }

    executor::SizeType32 maxNewTokens;
    if (!utils::extractSingleton<int32_t>(inputsTensors, InputFieldsNames::maxNewTokens, maxNewTokens))
    {
        throw std::runtime_error("request_output_len is not present in the request");
    }

    std::optional<executor::SizeType32> endId{std::nullopt};
    utils::extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::endId, endId);

    std::optional<executor::SizeType32> padId{std::nullopt};
    utils::extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::padId, padId);

    if (streaming && !isDecoupled)
    {
        throw std::runtime_error(
            "Streaming is only supported if model is "
            "deployed using decoupled mode.");
    }

    auto samplingConfig = utils::getSamplingConfigFromTensors(inputsTensors);

    std::optional<std::list<executor::VecTokens>> badWords = std::nullopt;
    executor::VecTokens badWordsRaw;
    if (utils::extractVector<int32_t>(inputsTensors, InputFieldsNames::badWords, badWordsRaw))
    {
        badWords = utils::convertWordList(badWordsRaw);
    }

    std::optional<std::list<executor::VecTokens>> stopWords = std::nullopt;
    executor::VecTokens stopWordsRaw;
    if (utils::extractVector<int32_t>(inputsTensors, InputFieldsNames::stopWords, stopWordsRaw))
    {
        stopWords = utils::convertWordList(stopWordsRaw);
    }

    std::optional<executor::Tensor> embeddingBias{std::nullopt};
    if (inputsTensors.count(InputFieldsNames::embeddingBias))
    {
        std::shared_ptr<runtime::ITensor> originalTensor = inputsTensors.at(InputFieldsNames::embeddingBias).tensor;
        utils::squeezeTensor(originalTensor, 1);
        auto newShape = originalTensor->getShape();
        if (!(newShape.nbDims == 1 && newShape.d[0] == 0))
        {
            embeddingBias = executor::detail::ofITensor(originalTensor);
        }
    }

    auto pTuningConfig = utils::getPromptTuningConfigFromTensors(inputsTensors);

    auto loraConfig = utils::getLoraConfigFromTensors(inputsTensors);

    auto speculativeDecodingConfig = utils::getSpeculativeDecodingConfigFromTensors(inputsTensors);

    return executor::Request(inputTokens, maxNewTokens, streaming, samplingConfig, outConfig, endId, padId, badWords,
        stopWords, embeddingBias, speculativeDecodingConfig, pTuningConfig, loraConfig, std::nullopt);
}

} // namespace triton::backend::inflight_batcher_llm::utils
