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
#define _GLIBCXX_USE_CXX11_ABI 0

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_METRICS
#include "custom_metrics_reporter/custom_metrics_reporter.h"
#endif

using namespace ::triton::common; // TritonJson

namespace triton::backend::inflight_batcher_llm
{

// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.

class ModelState
{
public:
    static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);

    template <typename T>
    T GetParameter(const std::string& name)
    {
        assert(false);
        auto dummy = T();
        return dummy;
    }

    virtual ~ModelState() = default;

#ifdef TRITON_ENABLE_METRICS
    TRITONSERVER_Error* InitCustomMetricsReporter(
        const std::string& model_name, const uint64_t version, const bool is_v1_model);
    TRITONSERVER_Error* UpdateCustomMetrics(const std::string& statistics);
#endif
    common::TritonJson::Value& GetModelConfig();

private:
#ifdef TRITON_ENABLE_METRICS
    std::unique_ptr<custom_metrics_reporter::CustomMetricsReporter> custom_metrics_reporter_;
#endif
    common::TritonJson::Value model_config_;
    std::shared_ptr<nvinfer1::ILogger> mTrtLogger{};

    ModelState(TRITONBACKEND_Model* triton_model, TritonJson::Value&& model_config)
        : model_config_(std::move(model_config))
    {
        mTrtLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
        initTrtLlmPlugins(mTrtLogger.get());
#ifdef TRITON_ENABLE_METRICS
        custom_metrics_reporter_ = std::make_unique<custom_metrics_reporter::CustomMetricsReporter>();
#endif
    }
};

template <>
std::string ModelState::GetParameter<std::string>(const std::string& name);

template <>
int32_t ModelState::GetParameter<int32_t>(const std::string& name);

template <>
uint32_t ModelState::GetParameter<uint32_t>(const std::string& name);

template <>
int64_t ModelState::GetParameter<int64_t>(const std::string& name);

template <>
uint64_t ModelState::GetParameter<uint64_t>(const std::string& name);

template <>
float ModelState::GetParameter<float>(const std::string& name);

template <>
bool ModelState::GetParameter<bool>(const std::string& name);

} // namespace triton::backend::inflight_batcher_llm
