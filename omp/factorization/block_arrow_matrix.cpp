// #include <ginkgo/core/factorization/block_arrow_lu.hpp>
// #include <ginkgo/core/base/types.hpp>
// #include "../../core/factorization/block_arrow_matrix.hpp"

// namespace gko {
// namespace kernels {
// namespace factorization {
// namespace block_arrow_lu {


// // col-major ordering
// template <typename ValueType, typename IndexType>
// void upper_triangular_left_solve_kernel(dim<2> dim_l_factor, const ValueType*
// l_factor,
//                                         dim<2> dim_lhs, ValueType*
//                                         lhs_matrix)
// {
//     for (IndexType col = 0; col < dim_l_factor[1]; col++) {
//         for (IndexType intern_index = 0; intern_index < col; intern_index++)
//         {
//             for (IndexType lhs_index = 0; lhs_index < dim_lhs[0];
//             lhs_index++) {
//                 lhs_matrix[dim_lhs[0] * col + lhs_index] -=
//                     l_factor[dim_l_factor[0] * col + intern_index] *
//                     lhs_matrix[dim_lhs[0] * intern_index + lhs_index];
//             }
//         }

//         // divide with pivot
//         auto pivot = l_factor[dim_l_factor[0] * col + col];
//         for (auto lhs_index = 0; lhs_index < (IndexType)dim_lhs[0];
//         lhs_index++) {
//             lhs_matrix[dim_lhs[0] * col + lhs_index] =
//                 lhs_matrix[dim_lhs[0] * col + lhs_index] / pivot;
//         }
//     }
// }

// template <typename ValueType, typename IndexType>
// void submatrix_11<ValueType, IndexType>::factorize(
//         std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
//         Partitions<IndexType>& partitions)
// {
//     using dense = matrix::Dense<ValueType>;
//     auto exec = global_mtx->get_executor();
//     const auto partition_idxs = partitions.get_const_data();
//     auto row_ptrs = row_ptrs_tmp.get_data();
//     const auto col_idxs = global_mtx->get_const_col_idxs();
//     const auto values = global_mtx->get_const_values();
//     exec->copy(split_index + 1, global_mtx->get_row_ptrs(), row_ptrs);
//     #pragma omp parallel for schedule(dynamic, 1)
//     for (auto block = 0; block < num_blocks; ++block) {
//         const auto stride = 1;
//         const auto len = static_cast<size_type>(partition_idxs[block + 1] -
//                                                 partition_idxs[block]);
//         const auto num_elems_dense = static_cast<size_type>(len * len);
//         dim<2> block_size = {len, len};

//         #pragma omp atomic
//         nnz_l_factor += (len * len + len) / 2;

//         #pragma omp atomic
//         nnz_u_factor += (len * len + len) / 2;

//         array<ValueType> values_tmp = {exec, num_elems_dense};
//         values_tmp.fill(0.0);
//         dense_diagonal_blocks.get_data()[block] =
//             dense::create(exec, block_size, std::move(values_tmp), stride);
//         auto dense_l_factor = dense_l_factors.get_data()[block].get();
//         auto dense_u_factor = dense_u_factors.get_data()[block].get();
//         auto dense_block = dense_diagonal_blocks.get_data()[block].get();
//         auto row_start = partition_idxs[block];
//         auto row_end = partition_idxs[block + 1];
//         convert_csr_2_dense<ValueType, IndexType>(
//             block_size, row_ptrs, col_idxs, values, dense_block, row_start,
//             row_end);
//         factorize_kernel(dense_block, dense_l_factor, dense_u_factor);
//      }
// }

// template <typename ValueType, typename IndexType>
// void submatrix_11<ValueType, IndexType>::factorize(
//         std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
//         Partitions<IndexType>& partitions)
//     {
//         using dense = matrix::Dense<ValueType>;
//         auto exec = global_mtx->get_executor();
//         const auto partition_idxs = partitions.get_const_data();
//         auto row_ptrs = row_ptrs_tmp.get_data();
//         const auto col_idxs = global_mtx->get_const_col_idxs();
//         const auto values = global_mtx->get_const_values();
//         exec->copy(split_index + 1, global_mtx->get_row_ptrs(), row_ptrs);
//         #pragma omp parallel for schedule(dynamic, 1)
//         for (auto block = 0; block < num_blocks; ++block) {
//             const auto stride = 1;
//             const auto len = static_cast<size_type>(partition_idxs[block + 1]
//             -
//                                                     partition_idxs[block]);
//             const auto num_elems_dense = static_cast<size_type>(len * len);
//             dim<2> block_size = {len, len};

//             #pragma omp atomic
//             nnz_l_factor += (len * len + len) / 2;

//             #pragma omp atomic
//             nnz_u_factor += (len * len + len) / 2;

//             array<ValueType> values_tmp = {exec, num_elems_dense};
//             values_tmp.fill(0.0);
//             dense_diagonal_blocks.get_data()[block] =
//                 dense::create(exec, block_size, std::move(values_tmp),
//                 stride);

//             auto dense_l_factor = dense_l_factors.get_data()[block].get();
//             auto dense_u_factor = dense_u_factors.get_data()[block].get();
//             auto dense_block = dense_diagonal_blocks.get_data()[block].get();
//             auto row_start = partition_idxs[block];
//             auto row_end = partition_idxs[block + 1];
//             convert_csr_2_dense<ValueType, IndexType>(
//                 block_size, row_ptrs, col_idxs, values, dense_block,
//                 row_start, row_end);
//             factorize_kernel(dense_block, dense_l_factor, dense_u_factor);
//          }
//     }

// template <typename ValueType, typename IndexType>
// void submatrix_12<ValueType, IndexType>::factorize(submatrix_11<ValueType,
// IndexType>& submtx_11,
//                 Partitions<IndexType>& partitions)
// {
//     auto block_row_ptrs_data = block_row_ptrs.get_data();
//     auto partition_idxs = partitions.array.get_data();
//     auto dense_l_factors = submtx_11.dense_l_factors;
//     using dense = matrix::Dense<ValueType>;
//     array<ValueType> residuals = array<ValueType>(exec, nz);
//     exec->copy<ValueType>(nz, mtx->get_values(), residuals.get_data());
//     #pragma omp parallel for schedule(dynamic)
//     for (IndexType block = 0; block < num_blocks; block++) {
//         if (nz_per_block.get_data()[block] > 0) {
//             auto stride = 1;
//             IndexType block_size =
//                 partition_idxs[block + 1] - partition_idxs[block];
//             dim<2> dim_tmp = {static_cast<size_type>(block_size),
//                                 static_cast<size_type>(block_size)};
//             dim<2> dim_rhs;
//             dim_rhs[0] = block_size;
//             dim_rhs[1] = (block_row_ptrs_data[block + 1] -
//                             block_row_ptrs_data[block]) /
//                             block_size;
//             auto values_l_factor =
//             dense_l_factors.get_data()[block].get()->get_values(); auto
//             values_12 = &mtx->get_values()[block_row_ptrs_data[block]];

//             lower_triangular_solve_kernel<ValueType, IndexType>(
//                 dim_tmp, values_l_factor, dim_rhs, values_12);

//             auto num_elems = dim_rhs[0] * dim_rhs[1];
//             auto values_residual =
//                 &residuals.get_data()[block_row_ptrs_data[block]];
//             auto residual_vectors = dense::create(
//                 exec, dim_rhs,
//                 array<ValueType>::view(exec, num_elems, values_residual),
//                 stride);

//             dim<2> dim_rnorm = {1, dim_rhs[1]};
//             array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
//             values_rnorm.fill(0.0);
//             auto residual_norm = dense::create(exec, dim_rnorm,
//                 values_rnorm, stride);

//             auto l_factor = share(dense::create(
//                 exec, dim_tmp,
//                 array<ValueType>::view(exec, dim_tmp[0] * dim_tmp[1],
//                                         values_l_factor), stride));

//             auto solution =
//                 dense::create(exec, dim_rhs,
//                                 array<ValueType>::view(
//                                     exec, dim_rhs[0] * dim_rhs[1],
//                                     values_12),
//                                 stride);

//             // mm multiplication
//             auto x = solution->get_values();
//             auto b = residual_vectors->get_values();
//             auto l_vals = l_factor->get_values();
//             for (auto row_l = 0; row_l < dim_tmp[0]; row_l++) {
//                 for (auto col_b = 0; col_b < dim_rhs[1]; col_b++) {
//                     for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
//                         b[dim_rhs[1]*row_l + col_b] -=
//                         l_vals[dim_tmp[1]*row_l + row_b]*x[dim_rhs[1]*row_b +
//                         col_b];
//                     }
//                 }
//             }

//             // compute residual norms
//             auto r = residual_vectors.get();
//             r->compute_norm2(residual_norm.get());
//             for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
//                 if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
//                     std::cout << "i: " << i << "abs values: " <<
//                     std::abs(residual_norm->get_values()[i]) << ",
//                     block_index: " << block << '\n'; break;
//                 }
//             }
//         }
//     }
// }

// template <typename ValueType, typename IndexType>
// void submatrix_21<ValueType, IndexType>::factorize(submatrix_11<ValueType,
// IndexType>& submtx_11,
//                 Partitions<IndexType>& partitions)
// {
//     auto block_col_ptrs_data = block_col_ptrs.get_data();
//     auto partition_idxs = partitions.array.get_data();
//     auto dense_u_factors = submtx_11.dense_u_factors;
//     using dense = matrix::Dense<ValueType>;
//     array<ValueType> residuals = array<ValueType>(exec, nz);
//     exec->copy<ValueType>(nz, mtx->get_values(), residuals.get_data());
//     #pragma omp parallel for schedule(dynamic)
//     for (auto block = 0; block < num_blocks; block++) {
//         if (nz_per_block.get_data()[block] > 0) {
//             auto stride = 1;
//             IndexType row_idx = partition_idxs[block];
//             IndexType block_size =
//                 partition_idxs[block + 1] - partition_idxs[block];
//             dim<2> dim_tmp = {static_cast<size_type>(block_size),
//                                 static_cast<size_type>(block_size)};

//             // this is reversed to act as a transposed solution
//             dim<2> dim_rhs;
//             dim_rhs[0] = (block_col_ptrs_data[block + 1] -
//             block_col_ptrs_data[block]) /
//                 block_size;
//             dim_rhs[1] = block_size;
//             auto values_u_factor =
//                 dense_u_factors.get_data()[block].get()->get_values();
//             auto values_21 = &mtx->get_values()[block_col_ptrs_data[block]];
//             upper_triangular_left_solve_kernel<ValueType, IndexType>(
//                 dim_tmp, values_u_factor, dim_rhs, values_21);

//             auto num_elems = dim_rhs[0] * dim_rhs[1];
//             auto values_residual =
//                 &residuals.get_data()[block_col_ptrs_data[block]];
//             auto residual_vectors = dense::create(
//                 exec, dim_rhs,
//                 array<ValueType>::view(exec, num_elems, values_residual),
//                 stride);
//             dim<2> dim_rnorm = {1, dim_rhs[1]};
//             array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
//             values_rnorm.fill(0.0);
//             auto residual_norm = dense::create(exec, dim_rnorm,
//                 values_rnorm, stride);

//             // careful stored in CSC format here so either transpose it or
//             use it as
//             // row vector x matrix operations
//             auto u_factor = share(dense::create(
//                 exec, dim_tmp,
//                 array<ValueType>::view(exec, dim_tmp[0] * dim_tmp[1],
//                                         values_u_factor), stride));
//             auto solution =
//                 dense::create(exec, dim_rhs,
//                                 array<ValueType>::view(
//                                     exec, dim_rhs[0] * dim_rhs[1],
//                                     values_21),
//                                 stride);

//             // mm multiplication (check if format is ok)
//             auto x = solution->get_values();
//             auto b = residual_vectors->get_values();
//             auto u_vals = u_factor->get_values();

//             for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
//                 for (auto col_l = 0; col_l < dim_tmp[1]; col_l++) {
//                     for (auto intern_index = 0; intern_index < dim_rhs[1];
//                     intern_index++) {
//                         b[row_b + dim_rhs[0]*col_l] -= u_vals[intern_index +
//                         dim_tmp[0]*col_l]*x[row_b + dim_rhs[0]*intern_index];
//                     }
//                 }
//             }

//             // compute residual norms
//             auto r = residual_vectors.get();
//             r->compute_norm2(residual_norm.get());
//             for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
//                 if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
//                     // std::cout << "  abs values: " <<
//                     std::abs(residual_norm->get_values()[i]) << '\n'; break;
//                 }
//             }
//         }
//     }
// }

// template<typename ValueType, typename IndexType>
// void submatrix_22<ValueType,
// IndexType>::compute_schur_complement_omp(submatrix_11<ValueType, IndexType>&
// submtx_11,
//                                 submatrix_12<ValueType, IndexType>&
//                                 submtx_12, submatrix_21<ValueType,
//                                 IndexType>& submtx_21, Partitions<IndexType>&
//                                 partitions)
// {
//     auto partition_idxs = partitions.array.get_data();
//     dim<2> size = {submtx_21.size[0], submtx_12.size[1]};
//     IndexType num_blocks = submtx_11.num_blocks;
//     #pragma omp parallel for schedule(dynamic)
//     for (auto block = 0; block < num_blocks; block++) {
//         auto blk = partition_idxs[block + 1] - partition_idxs[block];
//         spdgemm_blocks<ValueType, IndexType>(
//             size, blk, submtx_11, submtx_12, submtx_21, *this, block, -1);
//     }
// }

// }
// }
// }
// }