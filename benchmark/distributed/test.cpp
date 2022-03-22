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

#include <err.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"

int main(int argc, char* argv[])
{
    // Initialize the MPI execution environment
    MPI_Init(&argc, &argv);

    // Get the size of the group associated with communicator MPI_COMM_WORLD
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the calling process in the communicator MPI_COMM_WORLD
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                            MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(local_rank % num_devices);

    printf("* Allocate memory on [%1i],CPU\n", world_rank);
    int size = 1000;
    double* a = (double*)malloc(size * sizeof(double));
    if (a == NULL) {
        errx(1, "malloc a[] failed");
    }
    printf("* Allocate memory [%1i],GPU\n", world_rank);
    double* d_a;
    if (cudaMalloc((void**)&d_a, size * sizeof(double)) != cudaSuccess) {
        errx(1, "cudaMalloc d_a[] failed");
    }

    printf("* Initalize memory on [%1i],CPU\n", world_rank);
    for (int i = 0; i < size; i++) {
        a[i] = (double)world_rank;
    }

    MPI_Status status;
    int err;
    // From [0],CPU to [1],GPU
    if (world_rank == 0) {
        printf("* Send from [%1i],CPU\n", world_rank);
        err = MPI_Send(a, size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        printf("* Receive to [%1i],GPU\n", world_rank);
        err = MPI_Recv(d_a, size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
    if (err != MPI_SUCCESS) {
        errx(2, "MPI transport from [0],CPU to [1],GPU failed");
    }

    // From [1],GPU to [0],GPU
    if (world_rank == 1) {
        printf("* Send from [%1i],GPU\n", world_rank);
        err = MPI_Send(d_a, size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    } else if (world_rank == 0) {
        printf("* Receive to [%1i],GPU\n", world_rank);
        err = MPI_Recv(d_a, size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
    }
    if (err != MPI_SUCCESS) {
        errx(2, "MPI transport from [1],GPU to [0],GPU failed");
    }

    // From [0],GPU to [1],CPU
    if (world_rank == 0) {
        printf("* Send from [%1i],GPU\n", world_rank);
        err = MPI_Send(d_a, size, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        printf("* Receive to [%1i],CPU \n", world_rank);
        err = MPI_Recv(a, size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
    }
    if (err != MPI_SUCCESS) {
        errx(2, "MPI transport from [0],GPU to [1],CPU failed");
    }

    // From [1],CPU to [0],CPU
    if (world_rank == 1) {
        printf("* Send from [%1i],CPU\n", world_rank);
        err = MPI_Send(a, size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    } else if (world_rank == 0) {
        printf("* Receive to [%1i],CPU\n", world_rank);
        err = MPI_Recv(a, size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &status);
    }
    if (err != MPI_SUCCESS) {
        errx(2, "MPI transport from [1],CPU to [0],CPU failed");
    }

    // Check host memory
    for (int i = 0; i < size; i++) {
        if (a[i] != 0.) {
            errx(2, "MPI transport failed");
        }
    }

    printf("* Free memory on [%1i],GPU\n", world_rank);
    cudaFree(d_a);

    printf("* Free memory on [%1i],CPU\n", world_rank);
    free(a);

    // Terminates MPI execution environment
    MPI_Finalize();
}
