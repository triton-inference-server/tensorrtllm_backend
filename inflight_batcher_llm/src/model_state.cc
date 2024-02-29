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

#include "model_state.h"

namespace triton::backend::inflight_batcher_llm
{

/// Helper function to parse a csv delimited string to a vector ints
std::vector<int32_t> csvStrToVecInt(std::string const& str)
{
    std::vector<int32_t> output;
    std::stringstream ss(str);
    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        output.push_back(std::stoi(substr));
    }
    return output;
}

TRITONSERVER_Error* ModelState::Create(
    TRITONBACKEND_Model* triton_model, const std::string& name, const uint64_t version, ModelState** state)
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
        *state = new ModelState(triton_model, name, version, std::move(model_config));
    }
    catch (const std::exception& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

void ModelState::LoadParameters()
{
    // Check if model is in decoupled mode:
    triton::common::TritonJson::Value transaction_policy;
    model_config_.MemberAsObject("model_transaction_policy", &transaction_policy);
    transaction_policy.MemberAsBool("decoupled", &is_decoupled_);

    try
    {
        gpu_device_ids_ = GetParameter<std::vector<int32_t>>("gpu_device_ids");

        if (gpu_device_ids_)
        {
            std::string deviceIdInfo("Using GPU device ids: ");
            for (auto const& deviceId : gpu_device_ids_.value())
            {
                deviceIdInfo += std::to_string(deviceId) + " ";
            }
            TLLM_LOG_INFO(deviceIdInfo);
        }
    }
    catch (const std::exception& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("gpu_device_ids is not specified, will be automatically set");
    }
}

common::TritonJson::Value& ModelState::GetModelConfig()
{
    return model_config_;
}

const std::string& ModelState::GetModelName() const
{
    return model_name_;
}

uint64_t ModelState::GetModelVersion() const
{
    return model_version_;
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
std::vector<int32_t> ModelState::GetParameter<std::vector<int32_t>>(const std::string& name)
{
    auto deviceIdsStr = GetParameter<std::string>(name);
    // Parse as comma delimited string
    return csvStrToVecInt(deviceIdsStr);
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

} // namespace triton::backend::inflight_batcher_llm
