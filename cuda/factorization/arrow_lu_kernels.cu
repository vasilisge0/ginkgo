/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in src and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of src code must retain the above copyright
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

#include "core/factorization/cholesky_kernels.hpp"


#include <thrust/device_vector.h>
#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/arrow.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/sorting.cuh"
#include "cuda/components/thread_ids.cuh"

#include "core/factorization/arrow_lu_kernels.hpp"

namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace arrow_lu {

#include "common/cuda_hip/factorization/arrow_lu_kernels.hpp.inc"


// constexpr int default_block_size = 512;

//// Factorize kernels.
// template <typename ValueType, typename IndexType>
// void factorize_kernel(matrix::Dense<ValueType>* mtx,
//                      matrix::Dense<ValueType>* l_factor,
//                      matrix::Dense<ValueType>* u_factor)
//{
//    const auto mtx_values = mtx->get_values();
//    auto l_values = l_factor->get_values();
//    auto u_values = u_factor->get_values();
//    constexpr int subwarp_size = config::warp_size;
//    const auto block_dim = default_block_size;
//    constexpr auto block_size = default_block_size;
//    l_factor->fill(0.0);
//    u_factor->fill(0.0);
//
//    for (auto row = 0; row < mtx->get_size()[0]; ++row) {
//        const auto len = mtx->get_size()[0] - (row + 1);
//        const auto grid_dim =
//            static_cast<uint32>(ceildiv(len, block_dim / subwarp_size));
//
//        // Stores l_factor in row-major format.
//        update_l_factor_row<block_size, ValueType, IndexType>
//            <<<grid_dim, block_dim>>>(mtx_values, mtx->get_size()[0], row,
//            l_values);
//
//        // Stores u_factor in col-major format.
//        update_u_factor_col<block_size, ValueType, IndexType>
//            <<<grid_dim, block_dim>>>(mtx_values, mtx->get_size()[0], row,
//            u_values);
//    }
//}
//
////template <typename ValueType, typename IndexType>
////void step_1_impl_assemble(
////    const matrix::Csr<ValueType, IndexType>* global_mtx,
////    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
////    factorization::arrow_partitions<IndexType>& partitions)
/// GKO_NOT_IMPLEMENTED;
//
//// Step 1 for computing LU factors of submatrix_11. Initializes the dense
//// diagonal blocks of submatrix_11.
// template <typename ValueType, typename IndexType>
// void step_1_impl_assemble(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    using dense = matrix::Dense<ValueType>;
//    const auto exec = submtx_11.exec;
//    const size_type stride = 1;
//    const auto partition_idxs = partitions.data.get_data();
//    const auto num_blocks = submtx_11.num_blocks;
//
//    for (auto block = 0; block < num_blocks; block++) {
//        const dim<2> block_size = {static_cast<size_type>(partition_idxs[block
//        + 1] - partition_idxs[block]),
//                                   static_cast<size_type>(partition_idxs[block
//                                   + 1] - partition_idxs[block])};
//        auto tmp_array =
//            array<ValueType>(exec, block_size[0] * block_size[1]);
//        tmp_array.fill(0.0);
//        submtx_11.dense_l_factors.push_back(std::move(
//            dense::create(exec, block_size, std::move(tmp_array), stride)));
//    }
//
//    for (auto block = 0; block < num_blocks; block++) {
//        const dim<2> block_size = {static_cast<size_type>(partition_idxs[block
//        + 1] - partition_idxs[block]),
//                                   static_cast<size_type>(partition_idxs[block
//                                   + 1] - partition_idxs[block])};
//        auto tmp_array =
//            array<ValueType>(exec, block_size[0] * block_size[1]);
//        tmp_array.fill(0.0);
//        submtx_11.dense_u_factors.push_back(std::move(
//            dense::create(exec, block_size, std::move(tmp_array), stride)));
//    }
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_STEP_1_IMPL_ASSEMBLE_11_KERNEL);
//
//// Converts spare matrix in CSR format to dense.
// template <typename ValueType, typename IndexType>
// void convert_csr_2_dense(dim<2> size, const IndexType* row_ptrs,
//                         const IndexType* col_idxs, const ValueType* values,
//                         matrix::Dense<ValueType>* dense_mtx,
//                         const IndexType col_start, const IndexType col_end)
//{
//    auto values_mtx = dense_mtx->get_values();
//    const auto num_rows = dense_mtx->get_size()[0];
//    constexpr int subwarp_size = config::warp_size;
//    const auto block_dim = default_block_size;
//    const auto grid_dim =
//        static_cast<uint32>(ceildiv(size[0], block_dim / subwarp_size));
//    constexpr auto block_size = default_block_size;
//    convert_csr_2_dense_kernel<block_size, ValueType, IndexType>
//        <<<grid_dim, block_dim>>>(col_start, row_ptrs, col_idxs, values,
//            values_mtx, num_rows, col_end);
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL);
//
//
//// Step 2 for computing LU factors of submatrix_11. Computes the dense
//// LU factors of the diagonal blocks of submatrix_11.
// template <typename ValueType, typename IndexType>
// void step_2_impl_factorize(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    using dense = matrix::Dense<ValueType>;
//    const auto exec = submtx_11.exec;
//    const auto partition_idxs = partitions.get_const_data();
//    const auto col_idxs_src = global_mtx->get_const_col_idxs();
//    const auto values_src = global_mtx->get_const_values();
//    const auto stride = 1;
//    const auto split_index = submtx_11.split_index;
//    const auto row_ptrs_src = global_mtx->get_const_row_ptrs();
//    auto row_ptrs_submtx_11 = submtx_11.row_ptrs_tmp.get_data();
//    IndexType nnz_l_factor = 0;
//    IndexType nnz_u_factor = 0;
//    exec->copy(split_index + 1, row_ptrs_src, row_ptrs_submtx_11);
//
//    for (auto block = 0; block < submtx_11.num_blocks; block++) {
//        const auto len = static_cast<size_type>(partition_idxs[block + 1] -
//                                                partition_idxs[block]);
//        const auto num_elems_dense = static_cast<size_type>(len * len);
//        const dim<2> block_size = {len, len};
//        nnz_l_factor += (len * len + len) / 2;
//        nnz_u_factor += (len * len + len) / 2;
//        {
//            auto tmp_array =
//                array<ValueType>(exec, block_size[0] * block_size[1]);
//            tmp_array.fill(0.0);
//            auto tmp_dense_mtx =
//                dense::create(exec, block_size, std::move(tmp_array), stride);
//            submtx_11.dense_diagonal_blocks.push_back(std::move(tmp_dense_mtx));
//        }
//        auto dense_l_factor = submtx_11.dense_l_factors[block].get();
//        auto dense_u_factor = submtx_11.dense_u_factors[block].get();
//        auto dense_block = submtx_11.dense_diagonal_blocks[block].get();
//        const auto row_start = partition_idxs[block];
//        const auto row_end = partition_idxs[block + 1];
//        convert_csr_2_dense<ValueType, IndexType>(block_size,
//        row_ptrs_submtx_11,
//                                                  col_idxs_src, values_src,
//                                                  dense_block, row_start,
//                                                  row_end);
//        factorize_kernel(dense_block, dense_l_factor, dense_u_factor);
//    }
//    submtx_11.nnz_l_factor = nnz_l_factor;
//    submtx_11.nnz_u_factor = nnz_u_factor;
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_11_KERNEL);
//
//
////// Step 2 for computing LU factors of submatrix_11. Computes the dense
////// LU factors of the diagonal blocks of submatrix_11.
////template <typename ValueType, typename IndexType>
////void step_2_impl_factorize(
////    const matrix::Csr<ValueType, IndexType>* global_mtx,
////    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
////    factorization::arrow_partitions<IndexType>& partitions)
////{
////    using dense = matrix::Dense<ValueType>;
////    auto exec = submtx_11.exec;
////    const auto partition_idxs = partitions.get_const_data();
////    auto row_ptrs = submtx_11.row_ptrs_tmp.get_data();
////    const auto col_idxs = global_mtx->get_const_col_idxs();
////    const auto values = global_mtx->get_const_values();
////    const auto stride = 1;
////    IndexType nnz_l_factor = 0;
////    IndexType nnz_u_factor = 0;
////    exec->copy(submtx_11.split_index + 1, global_mtx->get_const_row_ptrs(),
////               row_ptrs);
////#pragma omp parallel for schedule(dynamic)
////    for (auto block = 0; block < submtx_11.num_blocks; block++) {
////        const auto len = static_cast<size_type>(partition_idxs[block + 1] -
////                                                partition_idxs[block]);
////        const auto num_elems_dense = static_cast<size_type>(len * len);
////        dim<2> block_size = {len, len};
////        nnz_l_factor += (len * len + len) / 2;
////        nnz_u_factor += (len * len + len) / 2;
////
////        {
////            auto tmp_array =
////                array<ValueType>(exec, block_size[0] * block_size[1]);
////            tmp_array.fill(0.0);
////            auto tmp =
////                dense::create(exec, block_size, std::move(tmp_array),
/// stride); / submtx_11.dense_diagonal_blocks.push_back(std::move(tmp)); / } /
/// auto dense_l_factor = submtx_11.dense_l_factors[block].get(); /        auto
/// dense_u_factor = submtx_11.dense_u_factors[block].get(); /        auto
/// dense_block = submtx_11.dense_diagonal_blocks[block].get(); /        auto
/// row_start = partition_idxs[block]; /        auto row_end =
/// partition_idxs[block + 1]; /        convert_csr_2_dense<ValueType,
/// IndexType>(block_size, row_ptrs, / col_idxs, values, dense_block, /
/// row_start, row_end); /        factorize_kernel(dense_block, dense_l_factor,
/// dense_u_factor); /    } /    submtx_11.nnz_l_factor = nnz_l_factor; /
/// submtx_11.nnz_u_factor = nnz_u_factor;
////}
////
////GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
////    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_11_KERNEL);
//
//// Step 1 of computing LU factors of submatrix_12. Computes the number of
//// nonzero entries of submatrix_12.
// template <typename ValueType, typename IndexType>
// void step_1_impl_symbolic_count(
//    std::shared_ptr<const DefaultExecutor> exec,
//    const matrix::Csr<ValueType, IndexType>* mtx,
//    factorization::arrow_submatrix_12<ValueType, IndexType>* submtx12,
//    factorization::arrow_partitions<IndexType>* partitions)
//{
//    auto size = submtx_12.size;
//    auto partition_idxs = partitions.get_data();
//    const auto row_ptrs_src = global_mtx->get_const_row_ptrs();
//    const auto col_idxs_src = global_mtx->get_const_col_idxs();
//    auto row_ptrs_current = submtx_12.row_ptrs_tmp.get_data();
//    auto block_row_ptrs = submtx_12.block_row_ptrs.get_data();
//    auto nz_per_block = submtx_12.nz_per_block.get_data();
//    exec->copy(submtx_12.size[0], row_ptrs_src, row_ptrs_current);
//    IndexType nnz_submatrix_12_count =
//        0;  // Number of nonzeros in submtx_12.mtx
//    IndexType col_min = 0;
//    IndexType row_min = 0;
//    IndexType num_occurences = 0;
//    const IndexType max_col = submtx_12.size[0] + submtx_12.size[1] + 1;
//    const auto num_blocks = submtx_12.num_blocks;
//    const auto split_index = submtx_12.split_index;
//    submtx_12.row_ptrs_tmp2 = {exec, size[0] + 1};
//    submtx_12.row_ptrs_tmp2.fill(0);
//    auto row_ptrs_submtx_12_current = submtx_12.row_ptrs_tmp2.get_data();
//    constexpr int subwarp_size = config::warp_size;
//
//    for (auto block = 0; block < num_blocks; block++) {
//        auto block_length = partitions.get_data()[block + 1] -
//        partitions.get_data()[block]; const auto block_dim =
//        default_block_size; const auto grid_dim =
//            static_cast<uint32>(ceildiv(block_length, block_dim /
//            subwarp_size));
//        constexpr auto block_size = default_block_size;
//        step_1_impl_symbolic_count_kernel<block_size, IndexType>
//            <<<grid_dim, block_dim>>>(submtx_12.row_ptrs_tmp.get_data(),
//            partitions.get_data(),
//                block, row_ptrs_current, row_ptrs_src, nz_per_block,
//                col_idxs_src, &nnz_submatrix_12_count, split_index, max_col,
//                block_row_ptrs);
//    }
//
//    // Synchronization here.
//    cudaDeviceSynchronize();
//
//    // Refreshes the row_ptrs_current. Nested parallelism here as well.
//    exec->copy(size[0], row_ptrs_src, row_ptrs_current);
//    const auto grid_dim =
//        static_cast<uint32>(ceildiv(size[0], default_block_size /
//        subwarp_size));
//    components::prefix_sum((std::shared_ptr<const
//    CudaExecutor>)submtx_12.mtx->get_executor(), row_ptrs_submtx_12_current,
//    size[0]);
//
//    // Updates the total nnz count of submatrix_12.
//    submtx_12.nz = nnz_submatrix_12_count;
//}
//
//
//// Step 2 of computing LU factors of submatrix_12. Initializes
//// the nonzero entries of submatrix_12.
// template <typename ValueType, typename IndexType>
// void step_2_impl_assemble(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    auto exec = submtx_12.exec;
//    auto nnzs = submtx_12.nz;
//    array<IndexType> col_idxs_tmp = {exec, static_cast<size_type>(nnzs)};
//    array<ValueType> values_tmp = {exec, static_cast<size_type>(nnzs)};
//    col_idxs_tmp.fill(0);
//    values_tmp.fill(0);
//    auto row_ptrs_current = submtx_12.row_ptrs_tmp.get_data();
//    auto partition_idxs = partitions.get_data();
//    auto values_src = global_mtx->get_const_values();
//    auto col_idxs_src = global_mtx->get_const_col_idxs();
//    auto row_ptrs_src = global_mtx->get_const_row_ptrs();
//    exec->copy(submtx_12.size[0], row_ptrs_src, row_ptrs_current);
//    auto values = values_tmp.get_data();
//    auto col_idxs = col_idxs_tmp.get_data();
//    auto row_ptrs = submtx_12.row_ptrs_tmp2.get_data();
//    auto num_blocks = submtx_12.num_blocks;
//    auto split_index = submtx_12.split_index;
//    std::vector<IndexType> row_ptrs_local;
//    IndexType max_row = submtx_12.size[0] + submtx_12.size[1] + 1;
//    constexpr int subwarp_size = config::warp_size;
//    const auto block_dim = default_block_size;
//    //constexpr auto block_size = default_block_size;
//
//    for (auto block = 0; block < num_blocks; block++) {
//        auto block_size = partition_idxs[block + 1] - partition_idxs[block];
//        auto row_start = partition_idxs[block];
//        auto row_end = partition_idxs[block + 1];
//        auto col_start = partition_idxs[block];
//        auto col_end = partition_idxs[block + 1];
//        IndexType num_occurences = 0;
//        // While there exists nonzero entries in block of submatrix_12.
//
//        const auto grid_dim =
//            static_cast<uint32>(ceildiv(block_size, block_dim /
//            subwarp_size));
//        while (1) {
//            IndexType col_min = 0;
//            IndexType row_min = 0;
//            IndexType remaining_nnz = 0;
//
//            // Find (row_min, col_min) s.t. col_min is the minimum column
//            // in the wavefront.
//            find_min_col_12_kernel<default_block_size, ValueType, IndexType>
//                <<<grid_dim, block_dim>>>(
//                    submtx_12.row_ptrs_tmp.get_data(),
//                    max_row,
//                    row_ptrs,
//                    col_idxs,
//                    row_start, row_end, &col_min, &row_min, &num_occurences);
//
//            // Update remaining_nnz for entries in block of global_mtx.
//            remaining_nnz =
//                symbolic_count_row_check((IndexType*)row_ptrs_current,
//                                         row_ptrs_src, row_start, row_end);
//            if (remaining_nnz == 0) {
//                break;
//            }
//
//            // Copies all nonzero entries in the wavefront of (1, 2) subblock
//            // iu global_mtx to submatrix_12.mtx.
//            update_row_ptrs_current_in_assemble_12<default_block_size,
//            ValueType, IndexType>
//                <<<grid_dim, block_dim>>>(col_idxs_src, values_src,
//                    row_ptrs_current, row_ptrs, col_idxs, values, col_min,
//                    split_index);
//        }
//    }
//
//    // @add_code --- Prefix sum here --- //
//    // Resets row_ptrs to original position.
//    //for (auto i = submtx_12.num_blocks - 1; i >= 0; i--) {
//    //    auto row_end = partition_idxs[i + 1];
//    //    auto row_start = partition_idxs[i];
//    //    for (auto j = row_end; j >= row_start; j--) {
//    //        row_ptrs[j] = (j > 0) ? row_ptrs[j - 1] : 0;
//    //    }
//    //}
//
//    // Creates submtx_12.mtx.
//    submtx_12.mtx = share(matrix::Csr<ValueType, IndexType>::create(
//        exec, submtx_12.size, std::move(values_tmp), std::move(col_idxs_tmp),
//        std::move(submtx_12.row_ptrs_tmp2)));
//}
//
//
//// Solves triangular system. L-factor is stored in row-major ordering.
// template <typename ValueType>
// void lower_triangular_solve_kernel(dim<2> dim_l_factor, ValueType* l_factor,
//                                   dim<2> dim_rhs, ValueType* rhs_matrix)
//{
//    // Computes rhs_matri[dim_rhs[1]*row + num_rhs] - dot_product[num_rhs]
//    //  = l_factor[row*dim_l_factor[1] + col]*rhs_matrix[col * dim_rhs[1] +
//    //  num_rhs]
//    // for all rows and col = 0, ..., row-1
//    for (auto row = 0; row < dim_l_factor[0]; row++) {
//        for (auto col = 0; col < row; col++) {
//            for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
//                rhs_matrix[dim_rhs[1] * row + num_rhs] -=
//                    l_factor[dim_l_factor[1] * row + col] *
//                    rhs_matrix[dim_rhs[1] * col + num_rhs];
//            }
//        }
//
//        // Computes (rhs_matri[dim_rhs[1]*row + num_rhs] - dot_product) /
//        pivot.
//        // Pivot = l_factor[dim_l_factor[0] * row + row];
//        auto pivot = l_factor[dim_l_factor[0] * row + row];
//        for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
//            rhs_matrix[dim_rhs[1] * row + num_rhs] =
//                rhs_matrix[dim_rhs[1] * row + num_rhs] / pivot;
//        }
//    }
//}
//
//// Step 3 of computing LU factors of submatrix_12. Sets up the
//// nonzero entries of submatrix_12 of U factor.
// template <typename ValueType, typename IndexType>
// void step_3_impl_factorize(
//    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
//    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    auto exec = submtx_11.exec;
//    auto block_row_ptrs_data = submtx_12.block_row_ptrs.get_data();
//    auto partition_idxs = partitions.get_data();
//    auto dense_l_factors = submtx_11.dense_l_factors.begin();
//    auto stride = 1;
//    auto num_blocks = submtx_12.num_blocks;
//    auto nz_per_block = submtx_12.nz_per_block.get_data();
//    using dense = matrix::Dense<ValueType>;
//    array<ValueType> residuals = array<ValueType>(exec, submtx_12.nz);
//    exec->copy(submtx_12.nz, submtx_12.mtx->get_values(),
//    residuals.get_data());
//
//    for (IndexType block = 0; block < num_blocks; block++) {
//        if (nz_per_block[block] > 0) {
//            auto block_size = static_cast<size_type>(partition_idxs[block + 1]
//            -
//                                                     partition_idxs[block]);
//            dim<2> dim_tmp = {static_cast<size_type>(block_size),
//                              static_cast<size_type>(block_size)};
//            dim<2> dim_rhs;
//            dim_rhs[0] = static_cast<size_type>(block_size);
//            dim_rhs[1] = static_cast<size_type>(
//                (block_row_ptrs_data[block + 1] - block_row_ptrs_data[block])
//                / block_size);
//
//            // Extracts values from dense block of submtx_11 and rhs in
//            // submtx_12 in CSR format.
//            auto values_l_factor =
//                submtx_11.dense_l_factors[block].get()->get_values();
//            auto values_12 =
//                &submtx_12.mtx->get_values()[block_row_ptrs_data[block]];
//            lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
//                                          values_12);
//
//            auto num_elems = dim_rhs[0] * dim_rhs[1];
//            auto values_residual =
//                &residuals.get_data()[block_row_ptrs_data[block]];
//            auto residual_vectors = dense::create(
//                exec, dim_rhs,
//                array<ValueType>::view(exec, num_elems, values_residual),
//                stride);
//
//            dim<2> dim_rnorm = {1, dim_rhs[1]};
//            array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
//            values_rnorm.fill(0.0);
//            auto residual_norm =
//                dense::create(exec, dim_rnorm, values_rnorm, stride);
//
//            auto l_factor = share(dense::create(
//                exec, dim_tmp,
//                array<ValueType>::view(exec, dim_tmp[0] * dim_tmp[1],
//                                       values_l_factor),
//                stride));
//
//            auto solution =
//                dense::create(exec, dim_rhs,
//                              array<ValueType>::view(
//                                  exec, dim_rhs[0] * dim_rhs[1], values_12),
//                              stride);
//
//            // Performs MM multiplication.
//            auto x = solution->get_values();
//            auto b = residual_vectors->get_values();
//            auto l_vals = l_factor->get_values();
//            for (auto row_l = 0; row_l < dim_tmp[0]; row_l++) {
//                for (auto col_b = 0; col_b < dim_rhs[1]; col_b++) {
//                    for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
//                        b[dim_rhs[1] * row_l + col_b] -=
//                            l_vals[dim_tmp[1] * row_l + row_b] *
//                            x[dim_rhs[1] * row_b + col_b];
//                    }
//                }
//            }
//
//            // Computes residual norms.
//            auto r = residual_vectors.get();
//            r->compute_norm2(residual_norm.get());
//            for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
//                if (abs(residual_norm->get_values()[i]) > 1e-8) {
//                    std::cout << "i: " << i << "abs values: ";
//                    std::cout << "block: " << block << '\n';
//                    break;
//                }
//            }
//        }
//    }
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_STEP_3_IMPL_FACTORIZE_12_KERNEL);
//
//
// template <typename ValueType, typename IndexType>
// void factorize_submatrix_11(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    step_1_impl_assemble(global_mtx, submtx_11, partitions);
//    step_2_impl_factorize(global_mtx, submtx_11, partitions);
//}
//
//// Updates entries of LU factorization of submatrix_12.
// template <typename ValueType, typename IndexType>
// void factorize_submatrix_12(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_partitions<IndexType>* partitions,
//    factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11,
//    factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12)
//{
//    //// Symbolic count and setup or row_ptrs.
//    //step_1_impl_symbolic_count(
//    //    glob,
//    //    submtx_12,
//    //    partitions);
//
//    //// Copies nonzero block-sized columns from blocks of (1, 2)-submatrix
//    //// of global_mtx to submtx_12.mtx in CSR format.
//    //step_2_impl_assemble(global_mtx, submtx_12, partitions);
//
//    //// Applies triangular solves on arrow_submtx_21 using the diagonal
//    //// blocks, stored in submtx_11. After the application of triangular
//    solves,
//    //// submtx_12.mtx contains the submatrix-(1, 2) of the U factor.
//    //step_3_impl_factorize(submtx_11, submtx_12, partitions);
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL);
//
//// Step 1 of computing LU factors of submatrix_21. Computes the number of
//// nonzeros of submatrix_21 of L factor.
// template <typename ValueType, typename IndexType>
// void step_1_impl_symbolic_count(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
//    factorization::arrow_partitions<IndexType>& partitions)
//{
//    const auto exec = submtx_21.exec;
//    const auto size = submtx_21.size;
//    const auto split_index = submtx_21.split_index;
//    const auto num_blocks = submtx_21.num_blocks;
//    const auto col_idxs_src = global_mtx->get_const_col_idxs();
//    const auto row_ptrs_src = global_mtx->get_const_row_ptrs();
//    const auto values_src = global_mtx->get_const_values();
//    const auto partition_idxs = partitions.get_data();
//    auto row_ptrs_current_src = submtx_21.row_ptrs_tmp.get_data();
//    auto block_col_ptrs_submtx_21 = submtx_21.block_col_ptrs.get_data();
//    submtx_21.block_col_ptrs.fill(0);
//    auto nz_per_block_submtx_21 = submtx_21.nz_per_block.get_data();
//    auto col_ptrs_submtx_21 = submtx_21.col_ptrs_tmp.get_data();
//    IndexType num_elems_src = 0;
//    IndexType max_num_elems_src = size[0] + size[1];
//
//    // Compressed representation of (row, row_ptrs).
//    std::vector<IndexType> compressed_rows_src;
//    std::vector<IndexType> compressed_row_ptrs_src;
//    array<IndexType> compressed_block_row_ptrs_src = {
//        exec, static_cast<size_type>(num_blocks) + 1};
//    compressed_block_row_ptrs_src.fill(0);
//    compressed_rows_src.resize(size[0] + size[1]);
//    compressed_row_ptrs_src.resize(size[0] + size[1]);
//    submtx_21.col_ptrs_tmp.fill(0);
//    exec->copy(size[0], &row_ptrs_src[split_index], row_ptrs_current_src);
//
//    // Main loop.
//    auto nz_count_total_submtx_21 = 0;
//    for (auto block = 0; block < num_blocks; block++) {
//        IndexType nz_count_submtx_21 = 0;
//        const auto col_start = partition_idxs[block];
//        const auto col_end = partition_idxs[block + 1];
//        const auto block_size =
//            partition_idxs[block + 1] - partition_idxs[block];
//        constexpr int subwarp_size = config::warp_size;
//        const auto block_dim = default_block_size;
//        const auto grid_dim =
//            static_cast<uint32>(ceildiv(size[0], block_dim / subwarp_size));
//        // This cannot be movec to gpu at this point.
//        for (auto row = 0; row < size[0]; row++) {
//            auto row_index_src = row_ptrs_current_src[row];
//            auto col_src = col_idxs_src[row_index_src];
//            // If current (row, col_src) entry remains in the current
//            // partition, increment nz_count_submtx_21 and update
//            // col_ptrs_submtx_21.
//            if ((col_src >= col_start) && (col_src < col_end)) {
//                nz_count_submtx_21 += block_size;
//
//                compressed_block_row_ptrs_src
//                    .get_data()[block + 1] += 1;
//
//                // std::cout << "row: " << row << ", nz_count_submtx_21: " <<
//                // nz_count_submtx_21 << '\n';
//                // If stored entries exceed size.
//                if (num_elems_src + 1 >= max_num_elems_src) {
//                    max_num_elems_src += (size[0] + size[1]);
//                    compressed_rows_src.resize(max_num_elems_src);
//                    compressed_row_ptrs_src.resize(
//                        max_num_elems_src);
//                }
//
//                // Stores.
//                compressed_rows_src[num_elems_src] =
//                    row + split_index;
//                compressed_row_ptrs_src[num_elems_src] =
//                    row_ptrs_current_src[row];
//                num_elems_src += 1;
//                row_ptrs_current_src[row] += 1;
//                row_index_src = row_ptrs_current_src[row];
//                col_src = col_idxs_src[row_index_src];
//
//                // Increment col_ptrs_submtx_21.
//                for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
//                     col_submtx_21++) {
//                    col_ptrs_submtx_21[col_submtx_21 + 1] += 1;
//                }
//
//                // Increments row_ptrs_current_src[row] until it reaches the
//                // beginning of the next block.
//                while ((col_src >= col_start) && (col_src < col_end)) {
//                    row_ptrs_current_src[row] += 1;
//                    row_index_src = row_ptrs_current_src[row];
//                    col_src = col_idxs_src[row_index_src];
//
//                    if (num_elems_src + 1 >= max_num_elems_src) {
//                        max_num_elems_src += (size[0] + size[1]);
//                        compressed_rows_src.resize(
//                            max_num_elems_src);
//                        compressed_row_ptrs_src.resize(
//                            max_num_elems_src);
//                    }
//
//                    // Stores (row, row_ptr).
//                    compressed_rows_src[num_elems_src] =
//                        row + split_index;
//                    compressed_row_ptrs_src[num_elems_src] =
//                        row_ptrs_current_src[row];
//                    num_elems_src += 1;
//                }
//            }
//        }
//        // Updates nonzero information for submtx_21.
//        nz_count_total_submtx_21 += nz_count_submtx_21;
//        // nz_count_total_submtx_21 +=
//        // compressed_block_row_ptrs_src.get_data()[block + 1];
//        nz_per_block_submtx_21[block] = nz_count_submtx_21;
//        block_col_ptrs_submtx_21[block + 1] = nz_count_total_submtx_21;
//    }
//
//    submtx_21.nz = nz_count_total_submtx_21;
//    // you have to set this
//
//    for (auto block = 0; block < num_blocks; block++) {
//        compressed_block_row_ptrs_src.get_data()[block + 1] +=
//            compressed_block_row_ptrs_src.get_data()[block];
//    }
//
//    {
//        array<IndexType> rows_in = {
//            exec, static_cast<size_type>(num_elems_src)};
//        array<IndexType> row_ptrs_in = {
//            exec, static_cast<size_type>(num_elems_src)};
//        array<IndexType> block_ptrs_in = {
//            exec, static_cast<size_type>(num_blocks) + 1};
//        for (auto i = 0; i < num_elems_src; i++) {
//            rows_in.get_data()[i] = compressed_rows_src[i];
//        }
//        for (auto i = 0; i < num_elems_src; i++) {
//            row_ptrs_in.get_data()[i] = compressed_row_ptrs_src[i];
//        }
//        for (auto i = 0; i < num_blocks + 1; i++) {
//            block_ptrs_in.get_data()[i] =
//                compressed_block_row_ptrs_src.get_data()[i];
//        }
//
//        submtx_21.block_storage =
//            std::make_shared<gko::factorization::block_csr_storage<IndexType>>(
//                rows_in, row_ptrs_in, block_ptrs_in);
//    }
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_21_KERNEL);
//
// template <typename ValueType, typename IndexType>
// void factorize_submatrix_21(
//    const matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::arrow_partitions<IndexType>& partitions,
//    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
//    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21)
//{
//    step_1_impl_symbolic_count(global_mtx, submtx_21, partitions);
//
//    // Allocates memory and assembles values of arrow_submatrix_21.
//    auto nnzs = static_cast<size_type>(submtx_21.nz);
//    auto exec = submtx_21.exec;
//    auto size = submtx_21.size;
//    dim<2> size_tmp = {size[1], size[0]};
//    auto col_ptrs_tmp = array<IndexType>(exec, size[1] + 1);
//    auto row_idxs_tmp = array<IndexType>(exec, nnzs);
//    auto values_tmp = array<ValueType>(exec, nnzs);
//    col_ptrs_tmp.fill(0);
//    row_idxs_tmp.fill(0);
//    values_tmp.fill(0.0);
//    submtx_21.mtx = share(matrix::Csr<ValueType, IndexType>::create(
//        exec, size_tmp, std::move(values_tmp), std::move(row_idxs_tmp),
//        std::move(col_ptrs_tmp)));
//    step_2_impl_assemble(global_mtx, submtx_21, partitions);
//
//    // applies triangular solves on arrow_submatrix_21 using the diagonal
//    // blocks, stored in arrow_submatrix_11.
//    step_3_impl_factorize(submtx_11, submtx_21, partitions);
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL);
//
// template <typename ValueType, typename IndexType>
// void compute_factors(
//    std::shared_ptr<const DefaultExecutor> exec,
//    gko::matrix::Csr<ValueType, IndexType>* global_mtx,
//    factorization::ArrowLuState<ValueType, IndexType>* workspace)
//{
//    factorize_submatrix_11(global_mtx, workspace->mtx_.submtx_11_,
//                           workspace->mtx_.partitions_);
//    factorize_submatrix_12(global_mtx, workspace->mtx_.partitions_.get(),
//                           workspace->mtx_.submtx_11_.get(),
//                           workspace->mtx_.submtx_12_.get());
//    factorize_submatrix_21(global_mtx, workspace->mtx_.partitions_,
//                           workspace->mtx_.submtx_11_,
//                           workspace->mtx_.submtx_21_);
//    factorize_submatrix_22(
//        global_mtx, workspace->mtx_.partitions_, workspace->mtx_.submtx_11_,
//        workspace->mtx_.submtx_12_, workspace->mtx_.submtx_21_,
//        workspace->mtx_.submtx_22_);
//}
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

template <typename ValueType, typename IndexType>
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::ArrowLuState<ValueType, IndexType>* workspace,
    const gko::matrix::Csr<ValueType, IndexType>* mtx) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

}  // namespace arrow_lu
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
