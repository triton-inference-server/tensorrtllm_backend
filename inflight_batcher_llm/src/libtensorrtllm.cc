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

#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <thread>

#include "tensorrt_llm/common/mpiUtils.h"

// Triton headers
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

// trtllm backend headers
#include "model_instance_state.h"
#include "model_state.h"
#include "orchestrator.h"
#include "work_item.h"
#include "work_items_queue.h"

#ifdef TRITON_ENABLE_METRICS
#include "custom_metrics_reporter/custom_metrics_reporter.h"
#endif

namespace triton::backend::inflight_batcher_llm
{

extern "C"
{

    // Global backend state creation
    TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
    {
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);

        char const* str = std::getenv("TRTLLM_ORCHESTRATOR");

        if (str && std::atoi(str) != 0)
        {
            TLLM_LOG_INFO(
                "Detected TRTLLM_ORCHESTRATOR environment variable, TRTLLM backend will operator in orchestrator "
                "mode.");
            auto* orchestrator = new Orchestrator();
            RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(orchestrator)));
        }
        else
        {
            RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(backend, nullptr));
        }

        return nullptr; // success
    }

    TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
    {
        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));

        if (vstate)
        {
            auto* orchestrator = reinterpret_cast<Orchestrator*>(vstate);
            delete orchestrator;
        }

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
    // to allow the backend to create any state associated with the model,
    // and to also examine the model configuration to determine if the
    // configuration is suitable for the backend. Any errors reported by
    // this function will prevent the model from loading.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
    {
        // Create a ModelState object and associate it with the
        // TRITONBACKEND_Model. If anything goes wrong with initialization
        // of the model state then an error is returned and Triton will fail
        // to load the model.
        char const* cname;
        RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
        const std::string name(cname);

        uint64_t version;
        RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

        ModelState* model_state;
        RETURN_IF_ERROR(ModelState::Create(model, name, version, &model_state));
        RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
    // needed. The backend should cleanup any state associated with the
    // model. This function will not be called until all model instances
    // of the model have been finalized.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
    {
        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
        delete model_state;

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
    // instance is created to allow the backend to initialize any state
    // associated with the instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
    {
        // Get the model state associated with this instance's model.
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        void* vmodelstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

        TRITONBACKEND_Backend* backend;
        RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));

        auto* orchestrator = reinterpret_cast<Orchestrator*>(vstate);

        if (orchestrator)
        {
            auto const device_ids = model_state->GetDeviceIds();
            int const num_workers = device_ids ? device_ids.value().size() : 1;

            std::string workerPath = model_state->GetWorkerPath();
            MPI_Comm everyone;
            MPI_Comm_spawn(workerPath.c_str(), MPI_ARGV_NULL, num_workers, MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone,
                MPI_ERRCODES_IGNORE);

            // The output comm is an intercommunicator so it has some special rules.
            // The parent must send data with bcast using root = MPI_ROOT (-4)
            std::vector<int64_t> packed = model_state->serialize();
            int64_t n = packed.size();
            MPICHECK(MPI_Bcast(&n, 1, MPI_INT64_T, MPI_ROOT, everyone));
            MPICHECK(MPI_Bcast(packed.data(), packed.size(), MPI_INT64_T, MPI_ROOT, everyone));

            OrchestratorCommunicator* communicator;
            RETURN_IF_ERROR(orchestrator->addCommunicator(model_state, instance, everyone, &communicator));
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(communicator)));
        }
        else
        {
            // Create a ModelInstanceState object and associate it with the
            // TRITONBACKEND_ModelInstance.
            ModelInstanceState* instance_state;
            RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));
        }

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
    // instance is no longer needed. The backend should cleanup any state
    // associated with the model instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
    {
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        TRITONBACKEND_Backend* backend;
        RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));

        auto* orchestrator = reinterpret_cast<Orchestrator*>(vstate);

        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));

        if (orchestrator)
        {
            auto* communicator = reinterpret_cast<OrchestratorCommunicator*>(vstate);
            communicator->shutdown();
            delete communicator;
        }
        else
        {
            ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);
            delete instance_state;
        }

        return nullptr; // success
    }

    // When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
    // that a backend create a response for each request in the batch. A
    // response may be the output tensors required for that request or may
    // be an error that is returned in the response.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
        TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
    {
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        TRITONBACKEND_Backend* backend;
        RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));

        auto* orchestrator = reinterpret_cast<Orchestrator*>(vstate);

        if (orchestrator)
        {
            OrchestratorCommunicator* communicator;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&communicator)));

            communicator->enqueue(requests, request_count);
        }
        else
        {
            ModelInstanceState* instance_state;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

            instance_state->enqueue(requests, request_count);
        }

        return nullptr; // success
    }

} // extern "C"

} // namespace triton::backend::inflight_batcher_llm
