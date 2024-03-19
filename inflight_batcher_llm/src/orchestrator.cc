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

#include "orchestrator.h"

#include "tensorrt_llm/common/mpiUtils.h"

#include "inference_answer.h"
#include "model_instance_state.h"
#include "utils.h"
#include "work_item.h"

namespace triton::backend::inflight_batcher_llm
{

OrchestratorCommunicator::OrchestratorCommunicator(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, MPI_Comm mpiComm)
    : model_state_(model_state)
    , modelInstance_(triton_model_instance)
{
    mWorkItemsQueue = std::make_unique<WorkItemsQueue>(isDecoupled());

    mMpiComm = std::make_unique<MpiComm>(mpiComm, true);

    mSenderThread = std::thread([this]() { SenderThread(); });
    mAnswerThread = std::thread([this]() { AnswerThread(); });
    mPollStopSignalThread = std::thread([this]() { PollStopSignalThread(); });
}

void OrchestratorCommunicator::SenderThread()
{
    while (true)
    {
        std::unique_lock lk(mSenderMutex);
        mSenderCV.wait(lk, [&]() { return (!mSenderQueue.empty()); });

        auto message = mSenderQueue.front();
        mSenderQueue.pop();

        if (message.id == MpiId::TERMINATION)
        {
            mMpiComm->send(&message.id, 1, MpiType::kUINT64, 0, kMPI_ID_TAG);
            TLLM_LOG_INFO("Orchestrator sender thread exiting");
            break;
        }
        else if (message.id == MpiId::PENDING_REQUEST)
        {
            auto& data = std::get<PendingRequestData>(message.data);

            std::vector<WorkItemsQueue::RequestWrapper> requestsToPush;
            std::vector<uint64_t> stopRequestIds;
            uint64_t exec_start_ns = 0;
            SET_TIMESTAMP(exec_start_ns);

            for (auto request : data.requests)
            {
                bool const isStopRequest
                    = utils::handleTritonRequest(request, mRequestIdStrMap, requestsToPush, *mWorkItemsQueue);

                if (isStopRequest)
                {
                    stopRequestIds.push_back(utils::getRequestId(request, mRequestIdStrMap));
                }
            }

            auto const workItemCb = [this, id = message.id](std::shared_ptr<WorkItem> wi)
            {
                auto packed = wi->getInferenceRequest()->serialize();

                mMpiComm->send(&id, 1, MpiType::kUINT64, 0, kMPI_ID_TAG);
                mMpiComm->send(packed.data(), packed.size(), MpiType::kINT64, 0, kMPI_DATA_TAG);
            };

            auto exceptions = mWorkItemsQueue->pushBatch(requestsToPush, exec_start_ns, workItemCb);

            if (!stopRequestIds.empty())
            {
                constexpr MpiId id = MpiId::STOP_REQUEST;
                mMpiComm->send(&id, 1, MpiType::kUINT64, 0, kMPI_ID_TAG);
                mMpiComm->send(stopRequestIds.data(), stopRequestIds.size(), MpiType::kUINT64, 0, kMPI_DATA_TAG);
            }
        }
        else if (message.id == MpiId::CANCEL_REQUEST)
        {
            auto& data = std::get<RequestIdsData>(message.data);

            mMpiComm->send(&message.id, 1, MpiType::kUINT64, 0, kMPI_ID_TAG);
            mMpiComm->send(data.ids.data(), data.ids.size(), MpiType::kUINT64, 0, kMPI_DATA_TAG);
        }
    }
}

void OrchestratorCommunicator::AnswerThread()
{
    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    MpiId mpiId;

    while (true)
    {
        mMpiComm->mprobe(0, kMPI_ID_TAG, &msg, &status);
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        TLLM_CHECK(count == 1);
        MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status));

        if (mpiId == MpiId::TERMINATION)
        {
            TLLM_LOG_INFO("Orchestrator answer thread exiting");
            break;
        }
        else if (mpiId == MpiId::REQUEST_IN_PROGRESS)
        {
            mMpiComm->mprobe(0, kMPI_DATA_TAG, &msg, &status);
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));

            std::vector<uint64_t> request_ids(count);
            MPICHECK(MPI_Mrecv(request_ids.data(), count, MPI_UINT64_T, &msg, &status));

            for (auto id : request_ids)
            {
                mWorkItemsQueue->markInProgress(id);
            }

            continue;
        }

        mMpiComm->mprobe(0, kMPI_DATA_TAG, &msg, &status);
        MPICHECK(MPI_Get_count(&status, MPI_INT64_T, &count));
        std::vector<int64_t> data(count);
        MPICHECK(MPI_Mrecv(data.data(), count, MPI_INT64_T, &msg, &status));

        auto answer = InferenceAnswer::deserialize(data.data());
        auto const requestId = answer->GetRequestId();

        std::string errStr = std::string("Failed to send Triton response for requestId: ")
            + utils::getRequestIdStr(requestId, mRequestIdStrMap);

        if (answer->IsFinalResponse())
        {
            mRequestIdStrMap.erase(requestId);
        }

        try
        {
            auto workItem = mWorkItemsQueue->getInProgressWorkItem(requestId);
            auto tritonErr = ModelInstanceState::sendTritonResponse(workItem, answer->GetTensors(),
                answer->IsFinalResponse(), answer->GetErrorMessage(), *mWorkItemsQueue, modelInstance_);
            LOG_IF_ERROR(tritonErr, errStr);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(errStr);
        }
    }
}

void OrchestratorCommunicator::PollStopSignalThread(int const intervalInMs)
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalInMs));

        if (mShutdownRequest.load())
        {
            break;
        }

        // Merge cancelled requests into stopped requests Ids
        auto cancelledReqIds = mWorkItemsQueue->getCancelledInProgressReqIds();

        if (cancelledReqIds.empty())
        {
            continue;
        }

        std::vector<uint64_t> cancelledReqIdsVec(cancelledReqIds.begin(), cancelledReqIds.end());

        MpiMessage message(MpiId::CANCEL_REQUEST);
        message.data = RequestIdsData{std::move(cancelledReqIdsVec)};

        SendMessage(std::move(message));
    }
}

void OrchestratorCommunicator::SendMessage(MpiMessage&& message)
{
    {
        std::unique_lock<std::mutex> lk(mSenderMutex);
        mSenderQueue.push(std::move(message));
    }

    mSenderCV.notify_all();
}

void OrchestratorCommunicator::enqueue(TRITONBACKEND_Request** requests, const uint32_t request_count)
{
    MpiMessage message(MpiId::PENDING_REQUEST);

    std::vector<TRITONBACKEND_Request*> data(requests, requests + request_count);
    message.data = PendingRequestData{std::move(data)};

    SendMessage(std::move(message));
}

void OrchestratorCommunicator::shutdown()
{
    MpiMessage message(MpiId::TERMINATION);

    {
        std::unique_lock<std::mutex> lk(mSenderMutex);
        mSenderQueue.push(message);
    }

    mSenderCV.notify_all();
    mShutdownRequest.store(true);

    if (mSenderThread.joinable())
    {
        mSenderThread.join();
    }
    if (mAnswerThread.joinable())
    {
        mAnswerThread.join();
    }
    if (mPollStopSignalThread.joinable())
    {
        mPollStopSignalThread.join();
    }
}

TRITONSERVER_Error* Orchestrator::addCommunicator(ModelState* model_state,
    TRITONBACKEND_ModelInstance* triton_model_instance, MPI_Comm mpiComm, OrchestratorCommunicator** communicator)
{
    *communicator = new OrchestratorCommunicator(model_state, triton_model_instance, mpiComm);

    {
        std::lock_guard<std::mutex> lk(mCommunicatorsMutex);
        mCommunicators.insert(*communicator);
    }

    return nullptr; // success
}

void Orchestrator::removeCommunicator(OrchestratorCommunicator* communicator)
{
    std::lock_guard<std::mutex> lk(mCommunicatorsMutex);
    mCommunicators.erase(communicator);
}

} // namespace triton::backend::inflight_batcher_llm
