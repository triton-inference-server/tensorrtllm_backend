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

#include "inference_answer.h"

namespace triton::backend::inflight_batcher_llm
{

static int kBitsinByte = 8;

std::vector<int64_t> InferenceAnswer::serialize() const
{
    std::list<int64_t> packed;

    packed.push_back(static_cast<int64_t>(request_id_));

    packed.push_back(static_cast<int64_t>(response_tensors_.size()));
    for (auto const& tensor : response_tensors_)
    {
        auto packed_tensor = tensor.serialize();
        packed.push_back(static_cast<int64_t>(packed_tensor.size()));
        packed.insert(packed.end(), packed_tensor.begin(), packed_tensor.end());
    }

    packed.push_back(final_response_ ? 1 : 0);

    const auto num_elements = (err_msg_.size() + sizeof(int64_t) - 1) / sizeof(int64_t);

    packed.push_back(static_cast<int64_t>(err_msg_.size()));

    for (size_t i = 0; i < num_elements; ++i)
    {
        int64_t buffer = 0;
        for (size_t j = 0; j < sizeof(int64_t) && (i * sizeof(int64_t) + j) < err_msg_.size(); ++j)
        {
            buffer |= static_cast<int64_t>(err_msg_[i * sizeof(int64_t) + j]) << (j * kBitsinByte);
        }
        packed.push_back(buffer);
    }

    std::vector<int64_t> vpacked{
        std::make_move_iterator(std::begin(packed)), std::make_move_iterator(std::end(packed))};
    return vpacked;
}

std::shared_ptr<InferenceAnswer> InferenceAnswer::deserialize(int64_t const* packed_ptr)
{
    auto const requestId = static_cast<uint64_t>(*packed_ptr++);
    auto answer = std::make_shared<InferenceAnswer>(requestId);

    int64_t num_tensors = *packed_ptr++;
    for (int64_t i = 0; i < num_tensors; ++i)
    {
        int64_t n{*packed_ptr++};
        auto tensor = NamedTensor::deserialize(packed_ptr);
        packed_ptr += n;
        answer->response_tensors_.push_back(std::move(tensor));
    }

    answer->final_response_ = *packed_ptr++ != 0;

    const auto num_chars = *packed_ptr++;
    answer->err_msg_.reserve(num_chars);

    int64_t i = 0;
    while (i < num_chars)
    {
        int64_t buffer = *packed_ptr++;
        for (size_t j = 0; j < sizeof(int64_t) && i < num_chars; ++i, ++j)
        {
            char c = static_cast<char>((buffer >> (j * kBitsinByte)) & 0xff);
            answer->err_msg_.push_back(c);
        }
    }

    return answer;
}

std::shared_ptr<InferenceAnswer> InferenceAnswer::deserialize(std::vector<int64_t> const& packed)
{
    return InferenceAnswer::deserialize(packed.data());
}

} // namespace triton::backend::inflight_batcher_llm
