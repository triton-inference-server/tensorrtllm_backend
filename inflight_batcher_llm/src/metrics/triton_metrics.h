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
#pragma once

#define _GLIBCXX_USE_CXX11_ABI 0
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace triton
{
namespace backend
{
namespace inflight_batcher_llm
{
namespace triton_metrics
{

class TritonMetricGroup
{
public:
    TritonMetricGroup(const std::string& metric_family_label, const std::string& metric_family_description,
        const std::string& category_label, std::vector<std::string>& json_keys, std::vector<std::string>& labels);
    ~TritonMetricGroup(){};
    TRITONSERVER_Error* CreateGroup(const std::string& model_name, const uint64_t version);
    TRITONSERVER_Error* UpdateGroup(std::vector<uint64_t>& values);
    const std::vector<std::string>& JsonKeys() const;

    struct MetricFamilyDeleter
    {
        void operator()(TRITONSERVER_MetricFamily* family)
        {
            if (family != nullptr)
            {
                TRITONSERVER_MetricFamilyDelete(family);
            }
        }
    };

    struct MetricDeleter
    {
        void operator()(TRITONSERVER_Metric* metric)
        {
            if (metric != nullptr)
            {
                TRITONSERVER_MetricDelete(metric);
            }
        }
    };

private:
    std::unique_ptr<TRITONSERVER_MetricFamily, MetricFamilyDeleter> metric_family_;
    std::vector<std::unique_ptr<TRITONSERVER_Metric, MetricDeleter>> metrics_;
    std::string metric_family_label_;
    std::string metric_family_description_;
    std::string category_label_;
    std::vector<std::string> json_keys_;
    std::vector<std::string> sub_labels_;
};

class TritonMetrics
{
public:
    TritonMetrics(){};
    ~TritonMetrics(){};
    // Update metrics for this backend.
    TRITONSERVER_Error* UpdateMetrics(const std::string& statistics);
    // Setup metrics for this backend.
    TRITONSERVER_Error* InitMetrics(const std::string& model, const uint64_t version, const bool is_v1_model);

private:
    std::vector<std::unique_ptr<TritonMetricGroup>> metric_groups_;

    /* Triton Metric Family Pointers */
    std::unique_ptr<TritonMetricGroup> request_metric_family_;
    std::unique_ptr<TritonMetricGroup> runtime_memory_metric_family_;
    std::unique_ptr<TritonMetricGroup> kv_cache_metric_family_;
    std::unique_ptr<TritonMetricGroup> model_type_metric_family_;

    // Timestamp and Iteration count metrics
    std::unique_ptr<TritonMetricGroup> general_metric_family_;
};

} // namespace triton_metrics
} // namespace inflight_batcher_llm
} // namespace backend
} // namespace triton
