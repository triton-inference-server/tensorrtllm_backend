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

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

// fwd declarations
struct TRITONBACKEND_Request;

namespace triton::backend::inflight_batcher_llm
{

// fwd declarations
class InferenceAnswer;

constexpr int32_t kMPI_ID_TAG{127};
constexpr int32_t kMPI_DATA_TAG{1023};

enum class MpiId : uint64_t
{
    PENDING_REQUEST = 1,
    REQUEST_IN_PROGRESS = 2,
    REQUEST_ANSWER = 3,
    STOP_REQUEST = 4,
    CANCEL_REQUEST = 5,
    TERMINATION = 6,
};

struct PendingRequestData
{
    std::vector<TRITONBACKEND_Request*> requests;
};

// Used by REQUEST_IN_PROGRESS and CANCEL_REQUEST
struct RequestIdsData
{
    std::vector<uint64_t> ids;
};

struct RequestAnswerData
{
    std::shared_ptr<InferenceAnswer> answer;
};

using MpiMessageData = std::variant<PendingRequestData, RequestIdsData, RequestAnswerData>;

struct MpiMessage
{
    MpiMessage(MpiId _id)
        : id(_id)
    {
    }

    MpiId id;

    MpiMessageData data;
};

} // namespace triton::backend::inflight_batcher_llm
