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

#include "tensorrt_llm/common/tllmException.h"
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "utils.h"

using namespace triton::backend::inflight_batcher_llm::utils;

TEST(UtilsTest, csvStrToVecInt)
{
    {
        std::string str = "0";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0));
    }
    {
        std::string str = "0, 1, 2";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    {
        std::string str = "0,1,2";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    // Trailing comma is ok
    {
        std::string str = "0, 1, 2, ";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    {
        std::vector<std::string> invalidStr{"a", "bbdfdsa", ",,", ","};
        for (auto const& str : invalidStr)
        {
            try
            {
                auto out = csvStrToVecInt(str);
                for (auto val : out)
                {
                    std::cout << val << std::endl;
                }
                FAIL() << "Expected exception for input: " << str;
            }
            catch (std::invalid_argument const& e)
            {
                EXPECT_THAT(e.what(), testing::HasSubstr("stoi"));
            }
            catch (tensorrt_llm::common::TllmException const& e)
            {
                EXPECT_THAT(
                    e.what(), testing::AnyOf(testing::HasSubstr("empty string"), testing::HasSubstr("Empty vector")));
            }
            catch (std::exception const& e)
            {
                FAIL() << "Expected invalid argument of input: " << str;
            }
        }
    }
}

TEST(UtilsTest, csvStrToVecVecInt)
{
    {
        std::string str = "{0}";
        auto out = csvStrToVecVecInt(str);
        EXPECT_THAT(out.size(), 1);
        EXPECT_THAT(out.at(0), testing::ElementsAre(0));
    }
    {
        std::vector<std::string> strs = {"{1, 5}, {1, 2, 3}", "{1,5},{1,2,3}", "{1, 5}, {1,2,3,}, "};
        for (auto str : strs)
        {
            auto out = csvStrToVecVecInt(str);
            EXPECT_THAT(out.size(), 2);
            EXPECT_THAT(out.at(0), testing::ElementsAre(1, 5));
            EXPECT_THAT(out.at(1), testing::ElementsAre(1, 2, 3));
        }
    }

    {
        std::vector<std::string> invalidStr{"a", "0,1,", "{a}", "{bdfda,,}", "", "", "{{}}", "{}"};
        for (auto const& str : invalidStr)
        {
            try
            {
                auto out = csvStrToVecVecInt(str);
                FAIL() << "Expected exception for input: " << str;
            }
            catch (std::invalid_argument const& e)
            {
                EXPECT_THAT(e.what(), testing::HasSubstr("stoi"));
            }
            catch (tensorrt_llm::common::TllmException const& e)
            {
                EXPECT_THAT(
                    e.what(), testing::AnyOf(testing::HasSubstr("empty string"), testing::HasSubstr("Empty vector")));
            }
            catch (std::exception const& e)
            {
                FAIL() << "Expected invalid argument of input: " << str << " got " << e.what();
            }
        }
    }
}
