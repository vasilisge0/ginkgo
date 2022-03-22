/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include <ginkgo/ginkgo.hpp>

#include <iostream>
#include <string>

#include <cuda_runtime.h>


#define cudaErrorCheck(call)                                         \
    do {                                                             \
        cudaError_t cudaErr = call;                                  \
        if (cudaSuccess != cudaErr) {                                \
            printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(cudaErr));                     \
            exit(0);                                                 \
        }                                                            \
    } while (0)


int get_node_local_rank(MPI_Comm comm)
{
    MPI_Comm local_comm;
    int rank = -1;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank);
    MPI_Comm_free(&local_comm);
    return rank;
}

int main(int argc, char* argv[])
{
    auto ompi_local_rank_c = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    auto ompi_local_rank = std::stoi(ompi_local_rank_c);
    // setenv("CUDA_VISIBLE_DEVICES", ompi_local_rank_c, 1);

    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto local_rank = get_node_local_rank(MPI_COMM_WORLD);
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    const char* gpu_id_list;
    const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if (cuda_visible_devices != nullptr) {
        gpu_id_list = cuda_visible_devices;
    } else {
        gpu_id_list = "N/A";
    }

    // Find how many GPUs HIP runtime says are available
    int num_devices = 0;
    cudaErrorCheck(cudaGetDeviceCount(&num_devices));

    int hwthread;
    int thread_id = 0;

    if (num_devices == 0) {
        hwthread = sched_getcpu();

        printf("MPI %03d (%03d|%03d) - OMP 000 - HWT %03d - Node %s\n", rank,
               local_rank, ompi_local_rank, hwthread, name);
    } else {
        char busid[64];
        std::string busid_list = "";
        std::string rt_gpu_id_list = "";

        // Loop over the GPUs available to each MPI rank
        for (int i = 0; i < std::min(num_devices, 7); i++) {
            cudaSetDevice(i);
            // Get the PCIBusId for each GPU and use it to query for UUID
            cudaDeviceGetPCIBusId(busid, 64, i);

            // Concatenate per-MPIrank GPU info into strings for print
            if (i > 0) rt_gpu_id_list.append(",");
            rt_gpu_id_list.append(std::to_string(i));

            std::string temp_busid(busid);

            if (i > 0) busid_list.append(",");
            busid_list.append(temp_busid.substr(5, 2));
        }

        hwthread = sched_getcpu();

        printf(
            "MPI %03d (%03d|%03d) - OMP 000 - HWT %03d - Node %s - RT_GPU_ID "
            "%s - "
            "GPU_ID %s - Bus_ID %s\n",
            rank, local_rank, ompi_local_rank, hwthread, name,
            rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
    }

    MPI_Finalize();
}
