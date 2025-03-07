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

#include "core/factorization/factorization_kernels.hpp"


#include <ginkgo/core/base/array.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


constexpr int default_block_size{512};


#include "common/cuda_hip/factorization/factorization_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const CudaExecutor> exec,
                           matrix::Csr<ValueType, IndexType>* mtx,
                           bool is_sorted)
{
    // TODO: Runtime can be optimized by choosing a appropriate size for the
    //       subwarp dependent on the matrix properties
    constexpr int subwarp_size = config::warp_size;
    auto mtx_size = mtx->get_size();
    auto num_rows = static_cast<IndexType>(mtx_size[0]);
    auto num_cols = static_cast<IndexType>(mtx_size[1]);
    size_type row_ptrs_size = num_rows + 1;
    if (num_rows == 0) {
        return;
    }

    array<IndexType> row_ptrs_addition(exec, row_ptrs_size);
    array<bool> needs_change_host{exec->get_master(), 1};
    needs_change_host.get_data()[0] = false;
    array<bool> needs_change_device{exec, 1};
    needs_change_device = needs_change_host;

    auto cuda_old_values = as_cuda_type(mtx->get_const_values());
    auto cuda_old_col_idxs = as_cuda_type(mtx->get_const_col_idxs());
    auto cuda_old_row_ptrs = as_cuda_type(mtx->get_row_ptrs());
    auto cuda_row_ptrs_add = as_cuda_type(row_ptrs_addition.get_data());

    const auto block_dim = default_block_size;
    const auto grid_dim =
        static_cast<uint32>(ceildiv(num_rows, block_dim / subwarp_size));
    if (is_sorted) {
        kernel::find_missing_diagonal_elements<true, subwarp_size>
            <<<grid_dim, block_dim>>>(
                num_rows, num_cols, cuda_old_col_idxs, cuda_old_row_ptrs,
                cuda_row_ptrs_add,
                as_cuda_type(needs_change_device.get_data()));
    } else {
        kernel::find_missing_diagonal_elements<false, subwarp_size>
            <<<grid_dim, block_dim>>>(
                num_rows, num_cols, cuda_old_col_idxs, cuda_old_row_ptrs,
                cuda_row_ptrs_add,
                as_cuda_type(needs_change_device.get_data()));
    }
    needs_change_host = needs_change_device;
    if (!needs_change_host.get_const_data()[0]) {
        return;
    }

    components::prefix_sum(exec, cuda_row_ptrs_add, row_ptrs_size);
    exec->synchronize();

    auto total_additions =
        exec->copy_val_to_host(cuda_row_ptrs_add + row_ptrs_size - 1);
    size_type new_num_elems = static_cast<size_type>(total_additions) +
                              mtx->get_num_stored_elements();


    array<ValueType> new_values{exec, new_num_elems};
    array<IndexType> new_col_idxs{exec, new_num_elems};
    auto cuda_new_values = as_cuda_type(new_values.get_data());
    auto cuda_new_col_idxs = as_cuda_type(new_col_idxs.get_data());

    // no empty kernel guard needed here, we exit earlier already
    kernel::add_missing_diagonal_elements<subwarp_size>
        <<<grid_dim, block_dim>>>(num_rows, cuda_old_values, cuda_old_col_idxs,
                                  cuda_old_row_ptrs, cuda_new_values,
                                  cuda_new_col_idxs, cuda_row_ptrs_add);

    const auto grid_dim_row_ptrs_update =
        static_cast<uint32>(ceildiv(num_rows, block_dim));
    kernel::update_row_ptrs<<<grid_dim_row_ptrs_update, block_dim>>>(
        num_rows + 1, cuda_old_row_ptrs, cuda_row_ptrs_add);

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs, IndexType* u_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};

    const auto block_size = default_block_size;
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size));
    const auto grid_dim = number_blocks;

    if (num_rows > 0) {
        kernel::count_nnz_per_l_u_row<<<grid_dim, block_size, 0, 0>>>(
            num_rows, as_cuda_type(system_matrix->get_const_row_ptrs()),
            as_cuda_type(system_matrix->get_const_col_idxs()),
            as_cuda_type(system_matrix->get_const_values()),
            as_cuda_type(l_row_ptrs), as_cuda_type(u_row_ptrs));
    }

    components::prefix_sum(exec, l_row_ptrs, num_rows + 1);
    components::prefix_sum(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* system_matrix,
                    matrix::Csr<ValueType, IndexType>* csr_l,
                    matrix::Csr<ValueType, IndexType>* csr_u)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    if (num_rows > 0) {
        kernel::initialize_l_u<<<grid_dim, block_size, 0, 0>>>(
            num_rows, as_cuda_type(system_matrix->get_const_row_ptrs()),
            as_cuda_type(system_matrix->get_const_col_idxs()),
            as_cuda_type(system_matrix->get_const_values()),
            as_cuda_type(csr_l->get_const_row_ptrs()),
            as_cuda_type(csr_l->get_col_idxs()),
            as_cuda_type(csr_l->get_values()),
            as_cuda_type(csr_u->get_const_row_ptrs()),
            as_cuda_type(csr_u->get_col_idxs()),
            as_cuda_type(csr_u->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};

    const auto block_size = default_block_size;
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size));
    const auto grid_dim = number_blocks;

    if (num_rows > 0) {
        kernel::count_nnz_per_l_row<<<grid_dim, block_size, 0, 0>>>(
            num_rows, as_cuda_type(system_matrix->get_const_row_ptrs()),
            as_cuda_type(system_matrix->get_const_col_idxs()),
            as_cuda_type(system_matrix->get_const_values()),
            as_cuda_type(l_row_ptrs));
    }

    components::prefix_sum(exec, l_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  matrix::Csr<ValueType, IndexType>* csr_l, bool diag_sqrt)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    if (num_rows > 0) {
        kernel::initialize_l<<<grid_dim, block_size, 0, 0>>>(
            num_rows, as_cuda_type(system_matrix->get_const_row_ptrs()),
            as_cuda_type(system_matrix->get_const_col_idxs()),
            as_cuda_type(system_matrix->get_const_values()),
            as_cuda_type(csr_l->get_const_row_ptrs()),
            as_cuda_type(csr_l->get_col_idxs()),
            as_cuda_type(csr_l->get_values()), diag_sqrt);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


}  // namespace factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
