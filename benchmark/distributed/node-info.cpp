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

#include <string>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/types.hpp"


int get_num_devices(std::shared_ptr<gko::Executor> exec)
{
    if (auto ptr = dynamic_cast<gko::CudaExecutor*>(exec.get())) {
        return ptr->get_num_devices();
    } else if (auto ptr = dynamic_cast<gko::HipExecutor*>(exec.get())) {
        return ptr->get_num_devices();
    } else if (auto ptr = dynamic_cast<gko::DpcppExecutor*>(exec.get())) {
        return std::max(ptr->get_num_devices("cpu"),
                        ptr->get_num_devices("gpu"));
    } else {
        return 0;
    }
}


std::string get_bus_id(std::shared_ptr<gko::Executor> exec)
{
    if (auto ptr = dynamic_cast<gko::CudaExecutor*>(exec.get())) {
        return ptr->get_bus_id();
    } else if (auto ptr = dynamic_cast<gko::HipExecutor*>(exec.get())) {
        return ptr->get_bus_id();
    } else if (auto ptr = dynamic_cast<gko::DpcppExecutor*>(exec.get())) {
        return ptr->get_bus_id();
    } else {
        return 0;
    }
}


int main(int argc, char* argv[])
{
    auto ompi_local_rank = std::stoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    // setenv("CUDA_VISIBLE_DEVICES", std::to_string(ompi_local_rank).c_str(),
    // 1);

    gko::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto local_rank = comm.node_local_rank();
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    std::string header = "";
    std::string format = "";
    initialize_argument_parsing(&argc, &argv, header, format);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm);

    const char* gpu_id_list;
    const char* rocr_visible_devices = std::getenv("ROCR_VISIBLE_DEVICES");
    const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        gpu_id_list = rocr_visible_devices;
    } else if (cuda_visible_devices != nullptr) {
        gpu_id_list = cuda_visible_devices;
    } else {
        gpu_id_list = "N/A";
    }

    // Find how many GPUs HIP runtime says are available
    int num_devices = get_num_devices(exec);

    int hwthread;
    int thread_id = 0;

#ifdef GINKGO_BUILD_OMP
    auto omp_wrapped = [](auto&& fn) {
#pragma omp parallel default(shared) private(hwthread, thread_id)
        {
#pragma omp critical
            {
                fn();
            }
        }
    };
#else
    auto omp_wrapped = [](auto&& fn) { fn(); };
#endif

    if (num_devices == 0) {
        omp_wrapped([&]() {
            hwthread = sched_getcpu();

            printf("MPI %03d (%03d|%03d) - OMP 000 - HWT %03d - Node %s\n",
                   rank, local_rank, ompi_local_rank, hwthread, name);
        });
    } else {
        std::string busid_list = "";
        std::string rt_gpu_id_list = "";

        // Loop over the GPUs available to each MPI rank
        for (int i = 0; i < num_devices; i++) {
            FLAGS_device_id = i;
            auto tmp_exec = executor_factory.at(FLAGS_executor)();

            // Concatenate per-MPIrank GPU info into strings for print
            if (i > 0) rt_gpu_id_list.append(",");
            rt_gpu_id_list.append(std::to_string(i));

            std::string temp_busid = get_bus_id(tmp_exec);

            if (i > 0) busid_list.append(",");
            busid_list.append(temp_busid.substr(5, 2));
        }

        omp_wrapped([&]() {
            hwthread = sched_getcpu();

            printf(
                "MPI %03d (%03d|%03d) - OMP 000 - HWT %03d - Node %s - "
                "RT_GPU_ID %s - "
                "GPU_ID %s - Bus_ID %s\n",
                rank, local_rank, ompi_local_rank, hwthread, name,
                rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
        });
    }
}
