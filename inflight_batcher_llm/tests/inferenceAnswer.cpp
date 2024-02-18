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

#include <gtest/gtest.h>

#include "tests/batch_manager/serializeDeserializeTestUtils.h"

#include "inference_answer.h"

#include "tensorrt_llm/common/logger.h"

using namespace triton::backend::inflight_batcher_llm;

class InferenceAnswerTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

template <typename T>
struct InferenceAnswerTestUtils
{
    static void CompareInferenceAnswers(InferenceAnswer& ia1, InferenceAnswer& ia2)
    {
        EXPECT_EQ(ia1.GetRequestId(), ia2.GetRequestId());
        EXPECT_EQ(ia1.GetTensors().size(), ia2.GetTensors().size());

        auto it1 = ia1.GetTensors().begin();
        auto it2 = ia2.GetTensors().begin();
        while (it1 != ia1.GetTensors().end() && it2 != ia2.GetTensors().end())
        {
            SerializeDeserializeTestUtils<T>::CompareTensors(*it1, *it2);
            ++it1;
            ++it2;
        }

        EXPECT_EQ(ia1.IsFinalResponse(), ia2.IsFinalResponse());
        EXPECT_EQ(ia1.GetErrorMessage(), ia2.GetErrorMessage());
    }

    static void TestInferenceAnswer(int requestId, bool finalAnswer, std::string& errMsg, nvinfer1::DataType dt)
    {
        // Force the creation of the logger so it gets destroyed last and can be used in the tensor destructor
        tensorrt_llm::common::Logger::getLogger();

        // create original InferenceAnswer
        std::list<NamedTensor> tensors;

        auto t1
            = SerializeDeserializeTestUtils<T>::CreateTensor(inference_request::kOutputIdsTensorName, {10, 20, 40}, dt);
        tensors.push_back(t1);

        auto t2 = SerializeDeserializeTestUtils<T>::CreateTensor(inference_request::kContextLogitsName, {6, 2}, dt);
        tensors.push_back(t2);
        auto ia1 = InferenceAnswer(requestId, tensors, finalAnswer, errMsg);

        // copy through deserialize(ia1.serialize())
        auto packed = ia1.serialize();
        auto ia2 = InferenceAnswer::deserialize(packed);

        InferenceAnswerTestUtils<T>::CompareInferenceAnswers(ia1, *ia2);
    }
};

TEST_F(InferenceAnswerTest, SerializeInt32)
{
    std::string emptyString;
    std::string errorMsg = "an error occurred";

    InferenceAnswerTestUtils<int32_t>::TestInferenceAnswer(12345, false, emptyString, nvinfer1::DataType::kINT32);
    InferenceAnswerTestUtils<int32_t>::TestInferenceAnswer(54321, false, emptyString, nvinfer1::DataType::kINT32);
    InferenceAnswerTestUtils<int32_t>::TestInferenceAnswer(12345, true, emptyString, nvinfer1::DataType::kINT32);
    InferenceAnswerTestUtils<int32_t>::TestInferenceAnswer(54321, true, errorMsg, nvinfer1::DataType::kINT32);
}

TEST_F(InferenceAnswerTest, SerializeInt8)
{
    std::string emptyString;
    std::string errorMsg = "an error occurred";

    InferenceAnswerTestUtils<int8_t>::TestInferenceAnswer(12345, false, emptyString, nvinfer1::DataType::kINT8);
    InferenceAnswerTestUtils<int8_t>::TestInferenceAnswer(54321, false, emptyString, nvinfer1::DataType::kINT8);
    InferenceAnswerTestUtils<int8_t>::TestInferenceAnswer(12345, true, emptyString, nvinfer1::DataType::kINT8);
    InferenceAnswerTestUtils<int8_t>::TestInferenceAnswer(54321, true, errorMsg, nvinfer1::DataType::kINT8);
}

TEST_F(InferenceAnswerTest, SerializeUInt8)
{
    std::string emptyString;
    std::string errorMsg = "an error occurred";

    InferenceAnswerTestUtils<uint8_t>::TestInferenceAnswer(12345, false, emptyString, nvinfer1::DataType::kUINT8);
    InferenceAnswerTestUtils<uint8_t>::TestInferenceAnswer(54321, false, emptyString, nvinfer1::DataType::kUINT8);
    InferenceAnswerTestUtils<uint8_t>::TestInferenceAnswer(12345, true, emptyString, nvinfer1::DataType::kUINT8);
    InferenceAnswerTestUtils<uint8_t>::TestInferenceAnswer(54321, true, errorMsg, nvinfer1::DataType::kUINT8);
}

TEST_F(InferenceAnswerTest, SerializeInt64)
{
    std::string emptyString;
    std::string errorMsg = "an error occurred";

    InferenceAnswerTestUtils<int64_t>::TestInferenceAnswer(12345, false, emptyString, nvinfer1::DataType::kINT64);
    InferenceAnswerTestUtils<int64_t>::TestInferenceAnswer(54321, false, emptyString, nvinfer1::DataType::kINT64);
    InferenceAnswerTestUtils<int64_t>::TestInferenceAnswer(12345, true, emptyString, nvinfer1::DataType::kINT64);
    InferenceAnswerTestUtils<int64_t>::TestInferenceAnswer(54321, true, errorMsg, nvinfer1::DataType::kINT64);
}

TEST_F(InferenceAnswerTest, SerializeFloat32)
{
    std::string emptyString;
    std::string errorMsg = "an error occurred";

    InferenceAnswerTestUtils<float>::TestInferenceAnswer(12345, false, emptyString, nvinfer1::DataType::kFLOAT);
    InferenceAnswerTestUtils<float>::TestInferenceAnswer(54321, false, emptyString, nvinfer1::DataType::kFLOAT);
    InferenceAnswerTestUtils<float>::TestInferenceAnswer(12345, true, emptyString, nvinfer1::DataType::kFLOAT);
    InferenceAnswerTestUtils<float>::TestInferenceAnswer(54321, true, errorMsg, nvinfer1::DataType::kFLOAT);
}
