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

#include "work_items_queue.h"
#include "work_item.h"

namespace triton::backend::inflight_batcher_llm
{

WorkItemsQueue::WorkItemsQueue(bool isDecoupled)
    : mIsDecoupled(isDecoupled)
{
}

void WorkItemsQueue::clear()
{
    std::lock_guard<std::mutex> lk(mMutex);
    mPendingWorkItems.clear();
    mPendingWorkItemsReqIds.clear();
    mInProgressWorkItems.clear();
    mStoppedReqIds.clear();
}

/// @brief Add a batch of new work item to the queue
/// Throws an error if requestId already exists
std::vector<std::shared_ptr<std::exception>> WorkItemsQueue::pushBatch(
    std::vector<RequestWrapper>& requestsToPush, uint64_t exec_start_ns)
{
    std::lock_guard<std::mutex> lk(mMutex);
    std::vector<std::shared_ptr<std::exception>> reqExceptions;
    for (auto& [requestId, request] : requestsToPush)
    {
        if (requestId != 0 && (hasInProgressReqId(requestId) || hasPendingReqId(requestId)))
        {
            std::string errStr
                = "requestId " + std::to_string(requestId) + " is already in progress, request is ignored.";
            reqExceptions.emplace_back(std::make_shared<std::runtime_error>(errStr));
        }
        else
        {
            try
            {
                auto workItem = requestId != 0 ? std::make_shared<WorkItem>(request, requestId, mIsDecoupled)
                                               : std::make_shared<WorkItem>(request, mIsDecoupled);
                mPendingWorkItems.push_back(workItem);
                mPendingWorkItemsReqIds.insert(workItem->requestId());
                workItem->getTimestamps().exec_start_ns = exec_start_ns;
                reqExceptions.push_back(nullptr);
            }
            catch (const std::exception& e)
            {
                reqExceptions.emplace_back(std::make_shared<std::runtime_error>(e.what()));
            }
        }
    }
    return reqExceptions;
}

std::tuple<std::shared_ptr<WorkItem>, bool> WorkItemsQueue::pop()
{
    std::lock_guard<std::mutex> lk(mMutex);

    if (mPendingWorkItems.empty())
    {
        return {nullptr, false};
    }

    auto workItem = mPendingWorkItems.front();
    mPendingWorkItems.pop_front();
    mPendingWorkItemsReqIds.erase(workItem->requestId());
    SET_TIMESTAMP(workItem->getTimestamps().compute_start_ns);

    // Check if work item has been stopped
    bool is_stopped = mStoppedReqIds.count(workItem->requestId());

    // Check if the Triton request has been cancelled
    bool is_cancelled = false;
    TRITONBACKEND_ResponseFactoryIsCancelled(workItem->response_factory(), &is_cancelled);

    bool stoppedRequest = false;
    if (!is_stopped && !is_cancelled)
    {
        mInProgressWorkItems.emplace(std::make_pair(workItem->requestId(), workItem));
    }
    else
    {
        mStoppedReqIds.erase(workItem->requestId());
        stoppedRequest = true;
    }

    return {workItem, stoppedRequest};
}

void WorkItemsQueue::markFinished(const uint64_t requestId)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (hasInProgressReqId(requestId))
    {
        mInProgressWorkItems.erase(requestId);
    }

    if (mStoppedReqIds.find(requestId) != mStoppedReqIds.end())
    {
        mStoppedReqIds.erase(requestId);
    }
}

void WorkItemsQueue::stopWorkItem(const uint64_t requestId)
{
    std::lock_guard<std::mutex> lk(mMutex);
    TLLM_LOG_DEBUG("Stopping request");
    if (hasInProgressReqId(requestId) || hasPendingReqId(requestId))
    {
        mStoppedReqIds.emplace(requestId);
    }
    else
    {
        std::string errStr = std::string("Received stop request for requestId ") + std::to_string(requestId)
            + std::string(" but it's not active (might be completed already).");
        throw std::runtime_error(errStr);
    }
}

std::unordered_set<uint64_t> WorkItemsQueue::getCancelledInProgressReqIds() const
{
    std::unordered_set<uint64_t> cancelledInProgressReqIds;
    {
        std::lock_guard<std::mutex> lk(mMutex);
        for (const auto& pair : mInProgressWorkItems)
        {
            bool is_cancelled = false;
            TRITONBACKEND_ResponseFactoryIsCancelled(pair.second->response_factory(), &is_cancelled);
            if (is_cancelled)
            {
                cancelledInProgressReqIds.emplace(pair.first);
            }
        }
    }
    return cancelledInProgressReqIds;
}

} // namespace triton::backend::inflight_batcher_llm
