#include "model_instance_state.h"

#include "tensorrt_llm/common/logger.h"

#include <mpi.h>

using namespace triton::backend::inflight_batcher_llm;

// This worker is launched from the TRT-LLM Triton backend when using the orchestrator mode
// It is intended to be a shim layer that instantiates a ModelInstanceObject that will
// communicate inference results back to the orchestrator in the Triton backend.
// In this application:
// - MPI_COMM_WORLD contains all workers participating in the model (one per GPU)
// - parentComm is an intercommunicator containing both MPI_COMM_WORLD and the process parent
// (i.e. a TRT-LLM Triton backend).
// See https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node198.htm#Node198 for
// more information on MPI inter-communicators
int main(int argc, char* argv[])
{
    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);

    MPI_Comm parentComm;
    MPI_Comm_get_parent(&parentComm);
    if (parentComm == MPI_COMM_NULL)
    {
        TLLM_LOG_ERROR("TRT-LLM worker has no parent!");
        return -1;
    }

    int size;
    MPI_Comm_remote_size(parentComm, &size);
    if (size != 1)
    {
        TLLM_LOG_ERROR("Parent size is %d, must be 1", size);
        return -1;
    }

    // TRT-LLM event synchronization takes extra time to complete
    // after the kernel has finished to run when using the spin wait
    // (default mode). Using a yield in the wait workarounds a large
    // part of the performance issue.
    TLLM_CUDA_CHECK(::cudaSetDeviceFlags(cudaDeviceScheduleYield));

    // Since parentComm is an intercommunicator, input root
    // is the rank of the parent process in his group
    // (always 0 as the parent size is checked before)
    int64_t packedSize;
    MPICHECK(MPI_Bcast(&packedSize, 1, MPI_INT64_T, 0, parentComm));
    std::vector<int64_t> packed(packedSize);
    MPICHECK(MPI_Bcast(packed.data(), packedSize, MPI_INT64_T, 0, parentComm));
    ModelState modelState = ModelState::deserialize(packed);

    TLLM_LOG_INFO("Worker loading model %s", modelState.GetModelName().c_str());

    ModelInstanceState* state;
    if (!ModelInstanceState::Create(&modelState, parentComm, &state))
    {
        return -1;
    }

    delete state;

    MPI_Finalize();
    return 0;
}
