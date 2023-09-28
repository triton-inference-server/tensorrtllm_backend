/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mpi.h>

#define MPICHECK(cmd)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

enum MpiType
{
    MPI_TYPE_BYTE,
    MPI_TYPE_CHAR,
    MPI_TYPE_INT,
    MPI_TYPE_INT64_T,
    MPI_TYPE_UINT32_T,
    MPI_TYPE_UINT64_T,
    MPI_TYPE_UNSIGNED_LONG_LONG,
};

inline MPI_Datatype getMpiDtype(MpiType dtype)
{
    static const std::unordered_map<MpiType, MPI_Datatype> dtype_map{
        {MPI_TYPE_BYTE, MPI_BYTE},
        {MPI_TYPE_CHAR, MPI_CHAR},
        {MPI_TYPE_INT, MPI_INT},
        {MPI_TYPE_INT64_T, MPI_INT64_T},
        {MPI_TYPE_UINT32_T, MPI_UINT32_T},
        {MPI_TYPE_UINT64_T, MPI_UINT64_T},
        {MPI_TYPE_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG},
    };
    return dtype_map.at(dtype);
}

inline int getCommWorldSize()
{
    int size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    return size;
}

inline int getCommWorldRank()
{
    int rank;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank;
}

inline void barrier()
{
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

inline void bcast(void* buffer, size_t size, MpiType dtype, int root)
{
    MPICHECK(MPI_Bcast(buffer, size, getMpiDtype(dtype), root, MPI_COMM_WORLD));
}

inline void bcast(std::vector<int64_t>& packed, int root)
{
    MPICHECK(MPI_Bcast(packed.data(), packed.size(), MPI_INT64_T, root, MPI_COMM_WORLD));
}
