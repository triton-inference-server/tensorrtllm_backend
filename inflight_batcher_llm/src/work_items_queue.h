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

#include "tensorrt_llm/common/logger.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "utils.h"
#include "work_item.h"
#include <list>

namespace triton::backend::inflight_batcher_llm
{

/// @brief Thread-safe queue of work items
class WorkItemsQueue
{
public:
    WorkItemsQueue(bool isDecoupled);

    /// @brief A wrapper for a request
    struct RequestWrapper
    {
        uint64_t request_id;
        TRITONBACKEND_Request* triton_request;

        // Enable implicit construction when using emplace_back()
        RequestWrapper(uint64_t id, TRITONBACKEND_Request* request)
            : request_id(id)
            , triton_request(request)
        {
        }
    };

    /// @brief Clear the queue
    void clear();

    // Note: this function only be called under a lock
    bool hasInProgressReqId(const uint64_t reqId) const
    {
        return (mInProgressWorkItems.find(reqId) != mInProgressWorkItems.end());
    }

    // Note: this function only be called under a lock
    bool hasPendingReqId(const uint64_t reqId) const
    {
        return (mPendingWorkItemsReqIds.find(reqId) != mPendingWorkItemsReqIds.end());
    }

    /// @brief Add a batch of new work item to the queue
    /// Throws an error if requestId already exists
    std::vector<std::shared_ptr<std::exception>> pushBatch(
        std::vector<RequestWrapper>& requestsToPush, uint64_t exec_start_ns);

    /// @brief Get a new work item from the queue, and move it to the list of
    /// in progress work items if it hasn't been stopped
    /// @return A tuple of the workItem and a boolean flag indicating if the work
    /// item has been marked in progress
    /// In case the queue is empty, return nullptr
    std::tuple<std::shared_ptr<WorkItem>, bool> pop();

    size_t numPendingWorkItems() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mPendingWorkItems.size();
    }

    std::shared_ptr<WorkItem> getInProgressWorkItem(uint64_t requestId)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mInProgressWorkItems.at(requestId);
    }

    /// @brief  Mark a request as being finished
    /// @param requestId
    void markFinished(const uint64_t requestId);

    // Stop a request by adding the request Id to a set
    // The set of stopped request id is used by the poll callback
    // and the pop function
    void stopWorkItem(const uint64_t requestId);

    std::unordered_set<uint64_t> getStoppedReqIds() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mStoppedReqIds;
    }

    std::unordered_set<uint64_t> getCancelledInProgressReqIds() const;

private:
    /// Queue of work items
    std::list<std::shared_ptr<WorkItem>> mPendingWorkItems;
    /// requestIds of work items in the queue
    std::set<uint64_t> mPendingWorkItemsReqIds;

    /// work items currently in progress
    std::unordered_map<uint64_t, std::shared_ptr<WorkItem>> mInProgressWorkItems;

    /// ids of the work items that have been stopped
    std::unordered_set<uint64_t> mStoppedReqIds;

    /// Whether model using this queue is decoupled
    bool mIsDecoupled;

    mutable std::mutex mMutex;
};

} // namespace triton::backend::inflight_batcher_llm
