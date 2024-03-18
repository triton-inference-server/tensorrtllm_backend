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

#include "model_state.h"
#include "mpi_utils.h"
#include "work_items_queue.h"

#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "tensorrt_llm/common/mpiUtils.h"

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>

using namespace tensorrt_llm::mpi;

namespace triton::backend::inflight_batcher_llm
{

class OrchestratorCommunicator
{
public:
    OrchestratorCommunicator(
        ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, MPI_Comm mpiComm);

    void enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count);
    void shutdown();

    bool isDecoupled() const
    {
        return model_state_->IsDecoupled();
    }

private:
    /// @brief Send work items to leader-worker ranks
    void SenderThread();
    /// @brief Receive inference answers from leader-worker ranks
    void AnswerThread();
    /// @brief Polls at a given interval for stop signals
    void PollStopSignalThread(int const invervalInMs = 10);

    void SendMessage(MpiMessage&& message);

private:
    ModelState* model_state_;
    TRITONBACKEND_ModelInstance* modelInstance_;

    std::unique_ptr<MpiComm> mMpiComm;

    std::unique_ptr<WorkItemsQueue> mWorkItemsQueue;

    std::thread mSenderThread;
    std::queue<MpiMessage> mSenderQueue;
    std::mutex mSenderMutex;
    std::condition_variable mSenderCV;

    std::thread mAnswerThread;
    std::thread mPollStopSignalThread;
    std::atomic<bool> mShutdownRequest = false;

    std::unordered_map<uint64_t, std::string> mRequestIdStrMap;
};

//
// Orchestrator
// Singleton class to track communicators
//

class Orchestrator
{
public:
    Orchestrator() {}

    virtual ~Orchestrator() {}

    TRITONSERVER_Error* addCommunicator(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
        MPI_Comm mpiComm, OrchestratorCommunicator** communicator);
    void removeCommunicator(OrchestratorCommunicator* communicator);

private:
    std::unordered_set<OrchestratorCommunicator*> mCommunicators;

    mutable std::mutex mCommunicatorsMutex;
};

} // namespace triton::backend::inflight_batcher_llm
