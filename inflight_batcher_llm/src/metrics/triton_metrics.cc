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
#include "triton_metrics.h"
#include "triton/backend/backend_common.h"
#include <vector>

#include <nlohmann/json.hpp>
using namespace ::triton::common; // TritonJson

namespace triton
{
namespace backend
{
namespace inflight_batcher_llm
{
namespace triton_metrics
{

uint64_t convertTimestampToMicroseconds(std::string* ts)
{
    std::tm tm = {};
    std::stringstream ss(*ts);
    ss >> std::get_time(&tm, "%m-%d-%Y %H:%M:%S");
    auto timestamp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    auto epoch = std::chrono::time_point_cast<std::chrono::microseconds>(timestamp).time_since_epoch();
    uint64_t time_in_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
    return time_in_microseconds;
}

TritonMetricGroup::TritonMetricGroup(const std::string& metric_family_label,
    const std::string& metric_family_description, const std::string& category_label,
    std::vector<std::string>& json_keys, std::vector<std::string>& sub_labels)
    : metric_family_label_(metric_family_label)
    , metric_family_description_(metric_family_description)
    , category_label_(category_label)
    , json_keys_(json_keys)
    , sub_labels_(sub_labels)
{
    // metric_family_(std::unique_ptr<TRITONSERVER_MetricFamily, MetricFamilyDeleter>(nullptr, MetricFamilyDeleter()))
}

TRITONSERVER_Error* TritonMetricGroup::CreateGroup(const std::string& model_name, const uint64_t version)
{
    TRITONSERVER_MetricFamily* metric_family = nullptr;
    RETURN_IF_ERROR(TRITONSERVER_MetricFamilyNew(&metric_family, TRITONSERVER_METRIC_KIND_GAUGE,
        metric_family_label_.c_str(), metric_family_description_.c_str()));
    metric_family_.reset(metric_family);
    std::vector<const TRITONSERVER_Parameter*> labels;
    labels.emplace_back(TRITONSERVER_ParameterNew("model", TRITONSERVER_PARAMETER_STRING, model_name.c_str()));
    labels.emplace_back(
        TRITONSERVER_ParameterNew("version", TRITONSERVER_PARAMETER_STRING, std::to_string(version).c_str()));

    for (size_t i = 0; i < sub_labels_.size(); i++)
    {
        TRITONSERVER_Metric* metric;
        labels.emplace_back(
            TRITONSERVER_ParameterNew(category_label_.c_str(), TRITONSERVER_PARAMETER_STRING, sub_labels_[i].c_str()));
        RETURN_IF_ERROR(TRITONSERVER_MetricNew(&metric, metric_family_.get(), labels.data(), labels.size()));
        std::unique_ptr<TRITONSERVER_Metric, MetricDeleter> unique_metric(metric);
        metrics_.push_back(std::move(unique_metric));
        TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(labels[2]));
        labels.pop_back();
    }

    for (const auto label : labels)
    {
        TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
    }

    return nullptr; // success
}

TRITONSERVER_Error* TritonMetricGroup::UpdateGroup(std::vector<uint64_t>& values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        RETURN_IF_ERROR(TRITONSERVER_MetricSet(metrics_[i].get(), values[i]));
    }
    return nullptr; // success
}

const std::vector<std::string>& TritonMetricGroup::JsonKeys() const
{
    return json_keys_;
}

TRITONSERVER_Error* TritonMetrics::InitMetrics(
    const std::string& model_name, const uint64_t version, const bool is_v1_model)
{
    /* REQUEST METRIC GROUP */
    std::vector<std::string> request_keys{"Active Request Count", "Max Request Count",
        "Scheduled Requests per Iteration", "Context Requests per Iteration"};
    std::vector<std::string> request_labels{"active", "max", "scheduled", "context"};
    request_metric_family_ = std::make_unique<TritonMetricGroup>(
        "nv_trt_llm_request_statistics", "TRT LLM request metrics", "request_type", request_keys, request_labels);

    RETURN_IF_ERROR(request_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(request_metric_family_));

    /* RUNTIME MEMORY METRIC GROUP */
    std::vector<std::string> runtime_memory_keys{
        "Runtime CPU Memory Usage", "Runtime GPU Memory Usage", "Runtime Pinned Memory Usage"};
    std::vector<std::string> runtime_memory_labels{"cpu", "gpu", "pinned"};
    runtime_memory_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_runtime_memory_statistics",
        "TRT LLM runtime memory metrics", "memory_type", runtime_memory_keys, runtime_memory_labels);

    RETURN_IF_ERROR(runtime_memory_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(runtime_memory_metric_family_));

    /* KV CACHE METRIC GROUP */
    std::vector<std::string> kv_cache_keys{
        "Max KV cache blocks", "Free KV cache blocks", "Used KV cache blocks", "Tokens per KV cache block"};
    std::vector<std::string> kv_cache_labels{"max", "free", "used", "tokens_per"};
    kv_cache_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_kv_cache_block_statistics",
        "TRT LLM KV cache block metrics", "kv_cache_block_type", kv_cache_keys, kv_cache_labels);

    RETURN_IF_ERROR(kv_cache_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(kv_cache_metric_family_));

    /* MODEL-TYPE METRIC GROUP (V1 / IFB) */
    std::vector<std::string> model_specific_keys{"Total Context Tokens per Iteration"};
    std::vector<std::string> model_specific_labels{"total_context_tokens"};
    std::string model = (is_v1_model) ? "v1" : "inflight_batcher";
    std::string model_metric_family_label = "nv_trt_llm_" + model + "_statistics";
    std::string model_metric_family_description = "TRT LLM " + model + "-specific metrics";
    std::string model_metric_family_category = model + "_specific_metric";

    if (is_v1_model)
    {
        model_specific_keys.push_back("Total Generation Tokens per Iteration");
        model_specific_labels.push_back("total_generation_tokens");
        model_specific_keys.push_back("Empty Generation Slots");
        model_specific_labels.push_back("empty_generation_slots");
    }
    else
    {
        model_specific_keys.push_back("Generation Requests per Iteration");
        model_specific_labels.push_back("generation_requests");
        model_specific_keys.push_back("MicroBatch ID");
        model_specific_labels.push_back("micro_batch_id");
    }

    model_type_metric_family_ = std::make_unique<TritonMetricGroup>(model_metric_family_label,
        model_metric_family_description, model_metric_family_category, model_specific_keys, model_specific_labels);

    RETURN_IF_ERROR(model_type_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(model_type_metric_family_));

    /* GENERAL METRIC GROUP */
    std::vector<std::string> general_metric_keys{"Timestamp", "Iteration Counter"};
    std::vector<std::string> general_metric_labels{"timestamp", "iteration_counter"};
    general_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_general_statistics",
        "General TRT LLM statistics", "general_type", general_metric_keys, general_metric_labels);

    RETURN_IF_ERROR(general_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(general_metric_family_));

    return nullptr; // success
}

TRITONSERVER_Error* TritonMetrics::UpdateMetrics(const std::string& statistics)
{
    triton::common::TritonJson::Value stats;
    std::vector<std::string> members;
    std::string timestamp;
    stats.Parse(statistics);
    stats.Members(&members);

    for (const auto& metric_group : metric_groups_)
    {
        std::vector<std::string> metric_group_keys = metric_group->JsonKeys();
        std::vector<uint64_t> metric_group_values;
        for (const auto& key : metric_group_keys)
        {
            triton::common::TritonJson::Value value_json;
            uint64_t value;
            stats.Find(key.c_str(), &value_json);
            if (key == "Timestamp")
            {
                value_json.AsString(&timestamp);
                value = convertTimestampToMicroseconds(&timestamp);
            }
            else
            {
                value_json.AsUInt(&value);
            }

            metric_group_values.push_back(value);
        }

        RETURN_IF_ERROR(metric_group->UpdateGroup(metric_group_values));
    }

    return nullptr;
}

} // namespace triton_metrics
} // namespace inflight_batcher_llm
} // namespace backend
} // namespace triton
