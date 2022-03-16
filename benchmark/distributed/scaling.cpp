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

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_int64(rows, 100,
             "Number of rows, either in total (strong_scaling == true) or per "
             "process (strong_scaling == false).");
DEFINE_int32(dim, 2, "Dimension of stencil, either 2D or 3D");
DEFINE_bool(restrict, false,
            "If true creates 5/7pt stencil, if false creates 9/27pt stencil.");
DEFINE_bool(strong_scaling, false,
            "If true benchmarks strong scaling, otherwise weak scaling.");
DEFINE_bool(graph_comm, false,
            "If true, the matrix will use neighborhood communication.");


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil(
    const gko::size_type global_size, const IndexType local_rows_start,
    const IndexType local_rows_end, bool restricted)
{
    const auto dp =
        static_cast<IndexType>(std::ceil(std::pow(global_size, 1. / 2.)));
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    for (IndexType row = local_rows_start; row < local_rows_end; row++) {
        auto i = row / dp;
        auto j = row % dp;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                if (!restricted || (d_i == 0 || d_j == 0)) {
                    auto col = j + d_j + (i + d_i) * dp;
                    if (col >= 0 && col < global_size) {
                        A_data.nonzeros.emplace_back(row, col,
                                                     gko::one<ValueType>());
                    }
                }
            }
        }
    }

    return A_data;
}


/**
 * Generates matrix data for a 3D stencil matrix. If restricted is set to true,
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil(
    const gko::size_type global_size, const IndexType local_rows_start,
    const IndexType local_rows_end, bool restricted)
{
    const auto dp =
        static_cast<IndexType>(std::ceil(std::pow(global_size, 1. / 2.)));
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    for (IndexType row = local_rows_start; row < local_rows_end; row++) {
        auto i = row / (dp * dp);
        auto j = (row % (dp * dp)) / dp;
        auto k = row % dp;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                for (IndexType d_k : {-1, 0, 1}) {
                    if (!restricted ||
                        ((d_i == 0 && d_j == 0) || (d_i == 0 && d_k == 0) ||
                         (d_j == 0 && d_k == 0))) {
                        auto col =
                            k + d_k + (j + d_j) * dp + (i + d_i) * dp * dp;
                        if (col >= 0 && col < global_size) {
                            A_data.nonzeros.emplace_back(row, col,
                                                         gko::one<ValueType>());
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


int main(int argc, char* argv[])
{
    gko::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    using ValueType = etype;
    using GlobalIndexType = gko::int64;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx =
        gko::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;

    std::string header =
        "A benchmark for measuring the strong or weak scaling of Ginkgo's "
        "distributed SpMV\n";
    std::string format = "";
    initialize_argument_parsing(&argc, &argv, header, format);

    if (rank == 0) {
        std::string extra_information = "";
        print_general_information(extra_information);
    }

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm);

    const auto num_rows = FLAGS_rows;
    const auto dim = FLAGS_dim;
    const bool restricted = FLAGS_restrict;
    const bool strong_scaling = FLAGS_strong_scaling;
    const bool graph_commm = FLAGS_graph_comm;

    // Generate matrix data on each rank
    if (rank == 0) {
        std::cout << "Generating stencil matrix..." << std::endl;
    }
    auto global_size = static_cast<gko::size_type>(
        strong_scaling ? num_rows : num_rows * comm.size());
    auto part = gko::distributed::Partition<LocalIndexType, GlobalIndexType>::
        build_from_global_size_uniform(
            exec->get_master(), comm.size(),
            static_cast<GlobalIndexType>(global_size));

    auto A_data =
        dim == 2 ? generate_2d_stencil<ValueType, GlobalIndexType>(
                       global_size, part->get_range_bounds()[comm.rank()],
                       part->get_range_bounds()[comm.rank() + 1], restricted)
                 : generate_3d_stencil<ValueType, GlobalIndexType>(
                       global_size, part->get_range_bounds()[comm.rank()],
                       part->get_range_bounds()[comm.rank() + 1], restricted);

    // Build global matrix from local matrix data.
    auto h_A = dist_mtx::create(exec->get_master(), comm);
    auto A = dist_mtx::create(exec, comm);
    h_A->read_distributed(A_data, part.get(), part.get());
    A->copy_from(h_A.get());
    if (graph_commm) {
        A->use_neighbor_comm();
    }

    // Set up global vectors for the distributed SpMV
    if (rank == 0) {
        std::cout << "Setting up vectors..." << std::endl;
    }
    const auto local_size =
        static_cast<gko::size_type>(part->get_part_size(comm.rank()));
    auto x = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    x->fill(gko::one<ValueType>());
    auto b = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    b->fill(gko::one<ValueType>());


    auto timer = get_timer(exec, FLAGS_gpu_timer);
    IterationControl ic{timer};

    // Do a warmup run
    if (rank == 0) {
        std::cout << "Warming up..." << std::endl;
    }
    for (auto _ : ic.warmup_run()) {
        A->apply(lend(x), lend(b));
    }

    // Do and time the actual benchmark runs
    if (rank == 0) {
        std::cout << "Running benchmark..." << std::endl;
    }
    for (auto _ : ic.run()) {
        A->apply(lend(x), lend(b));
    }
    if (rank == 0) {
        std::cout << "DURATION: " << ic.compute_average_time() << "s"
                  << std::endl;
        std::cout << "ITERATIONS: " << ic.get_num_repetitions() << std::endl;
    }
}
