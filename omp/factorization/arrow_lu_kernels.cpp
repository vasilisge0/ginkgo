#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The ArrowLu namespace.
 *
 * @ingroup factor
 */
namespace arrow_lu {

template <typename ValueType, typename IndexType>
void copy_off_diagonal_block(IndexType split_index, gko::span row_span,
                             IndexType col_min, IndexType* remaining_row_nnz_a,
                             const ValueType* a_values,
                             const IndexType* a_col_idxs,
                             IndexType* a_cur_row_ptrs, ValueType* s_values,
                             IndexType* s_col_idxs, IndexType* s_cur_row_ptrs)
{
    auto remaining_nnz = *remaining_row_nnz_a;
    bool found_nz_col = false;
    for (auto row = row_span.begin(); row < row_span.end(); row++) {
        const auto row_index_a = a_cur_row_ptrs[row];
        const auto col = a_col_idxs[row_index_a];
        if (col == col_min) {
            found_nz_col = true;
            break;
        }
    }

    if (found_nz_col) {
        for (auto row = row_span.begin(); row < row_span.end(); row++) {
            const auto row_index_a = a_cur_row_ptrs[row];
            const auto col = a_col_idxs[row_index_a];
            const auto row_index_s = s_cur_row_ptrs[row];
            s_values[row_index_s] =
                (col == col_min) ? a_values[row_index_a] : 0.0;
            s_col_idxs[row_index_s] =
                col_min - static_cast<IndexType>(split_index);
            s_cur_row_ptrs[row] += 1;
            if (col == col_min) {
                remaining_nnz -= 1;
                a_cur_row_ptrs[row] += 1;
            }
        }
    }
    *remaining_row_nnz_a = remaining_nnz;
}

template <typename IndexType>
void push_wavefront(dim<2> size, gko::span row_span, IndexType col_min,
                    IndexType* remaining_row_nnz_a, const IndexType* a_col_idxs,
                    const IndexType* a_row_ptrs, IndexType* a_cur_row_ptrs,
                    IndexType* nnz_s, IndexType* cur_s_row_ptrs)
{
    bool found_nonzero_column = false;
    auto nnz = *nnz_s;
    auto remaining_nnz = *remaining_row_nnz_a;
    auto len = row_span.end() - row_span.begin();
    for (auto row = row_span.begin(); row < row_span.end(); row++) {
        const auto row_index = a_cur_row_ptrs[row];
        const auto col = a_col_idxs[row_index];
        if ((col == col_min) && (a_cur_row_ptrs[row] < a_row_ptrs[row + 1])) {
            a_cur_row_ptrs[row] += 1;
            remaining_nnz -= 1;
            found_nonzero_column = true;
        }
    }

    if (found_nonzero_column) {
        for (auto row = row_span.begin(); row < row_span.end(); row++) {
            cur_s_row_ptrs[row + 1] += 1;
        }
    }
    const auto col_max = static_cast<IndexType>(size[0]);
    nnz = (found_nonzero_column && (col_min >= col_max)) ? nnz + len : nnz;
    *remaining_row_nnz_a = remaining_nnz;
    *nnz_s = nnz;
}

template <typename IndexType>
IndexType compute_remaining_nnz_row_check(gko::span row_span,
                                          const IndexType* row_ptrs,
                                          IndexType* cur_row_ptrs)
{
    auto remaining_nnz = 0;
    for (auto row = row_span.begin(); row < row_span.end(); row++) {
        remaining_nnz += ((row_ptrs[row + 1] > cur_row_ptrs[row])
                              ? row_ptrs[row + 1] - cur_row_ptrs[row]
                              : 0);
    }
    return remaining_nnz;
}

template <typename IndexType>
IndexType compute_remaining_nnz_col_check(gko::span row_span,
                                          gko::span col_span,
                                          const IndexType* col_idxs,
                                          IndexType* cur_row_ptrs)
{
    IndexType remaining_nnz = 0;
    for (auto row = row_span.begin(); row < row_span.end(); row++) {
        auto row_index = cur_row_ptrs[row];
        while ((col_idxs[row_index] < col_span.end()) &&
               (row_index < cur_row_ptrs[row + 1])) {
            cur_row_ptrs[row] += 1;
            row_index = cur_row_ptrs[row];
            remaining_nnz += 1;
        }
    }
    return remaining_nnz;
}

template <typename IndexType>
void find_min_col(dim<2> size, gko::span row_span, IndexType* col_min_out,
                  const IndexType* col_idxs, const IndexType* row_ptrs,
                  IndexType* cur_row_ptrs)
{
    const auto max_row = static_cast<IndexType>(size[0] + size[1]);
    auto col_min = max_row;
    IndexType num_occurences = 0;
    for (auto row = row_span.begin(); row < row_span.end(); row++) {
        const auto row_index = cur_row_ptrs[row];
        const auto col = col_idxs[row_index];
        col_min = ((col < col_min) && (row_index < row_ptrs[row + 1]))
                      ? col
                      : col_min;
    }
    *col_min_out = col_min;
}

// -- solve kernels --

// Solves triangular system. L-factor is stored in row-major ordering.
template <typename ValueType>
void lower_triangular_solve_kernel(dim<2> dim_l_factor,
                                   const ValueType* l_factor, dim<2> dim_rhs,
                                   ValueType* rhs_matrix)
{
    // Computes rhs_matri[dim_rhs[1]*row + num_rhs] - dot_product[num_rhs]
    //  = l_factor[row*dim_l_factor[1] + col]*rhs_matrix[col * dim_rhs[1] +
    //  num_rhs] for all rows and col = 0, ..., row-1
    for (auto row = 0; row < dim_l_factor[0]; row++) {
        for (auto col = 0; col < row; col++) {
            for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
                rhs_matrix[dim_rhs[1] * row + num_rhs] -=
                    l_factor[dim_l_factor[1] * row + col] *
                    rhs_matrix[dim_rhs[1] * col + num_rhs];
            }
        }

        // Computes (rhs_matri[dim_rhs[1]*row + num_rhs] - dot_product) / pivot.
        // Pivot = l_factor[dim_l_factor[0] * row + row];
        const auto pivot = l_factor[dim_l_factor[0] * row + row];
        for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
            rhs_matrix[dim_rhs[1] * row + num_rhs] =
                rhs_matrix[dim_rhs[1] * row + num_rhs] / pivot;
        }
    }
}

// Uses column-major ordering.
template <typename ValueType>
void upper_triangular_solve_kernel(dim<2> dim_l_factor,
                                   const ValueType* l_factor, dim<2> dim_rhs,
                                   ValueType* rhs_matrix)
{
    for (auto col = dim_l_factor[1] - 1; col >= 0; col--) {
        for (auto row = 0; row < col; row++) {
            for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
                rhs_matrix[row + dim_rhs[0] * num_rhs] -=
                    l_factor[row + dim_l_factor[0] * col] *
                    rhs_matrix[row + dim_rhs[0] * num_rhs];
            }
        }

        // Divides with pivot.
        auto pivot = l_factor[dim_l_factor[1] * col + col];
        for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
            rhs_matrix[col + dim_rhs[0] * num_rhs] =
                rhs_matrix[col + dim_rhs[0] * num_rhs] / pivot;
        }
    }
}

// Uses column-major ordering
template <typename ValueType>
void upper_triangular_left_solve_kernel(dim<2> dim_l_factor,
                                        const ValueType* l_factor,
                                        dim<2> dim_lhs, ValueType* lhs_matrix)
{
    for (auto col = 0; col < dim_l_factor[1]; col++) {
        for (auto intern_index = 0; intern_index < col; intern_index++) {
            for (auto lhs_index = 0; lhs_index < dim_lhs[0]; lhs_index++) {
                lhs_matrix[dim_lhs[0] * col + lhs_index] -=
                    l_factor[dim_l_factor[0] * col + intern_index] *
                    lhs_matrix[dim_lhs[0] * intern_index + lhs_index];
            }
        }

        // divide with pivot
        auto pivot = l_factor[dim_l_factor[0] * col + col];
        for (auto lhs_index = 0; lhs_index < dim_lhs[0]; lhs_index++) {
            lhs_matrix[dim_lhs[0] * col + lhs_index] =
                lhs_matrix[dim_lhs[0] * col + lhs_index] / pivot;
        }
    }
}

template <typename ValueType>
void compute_dense_lu_kernel(const matrix::Dense<ValueType>* mtx,
                             matrix::Dense<ValueType>* l_factor,
                             matrix::Dense<ValueType>* u_factor)
{
    const auto mtx_values = mtx->get_const_values();
    auto l_values = l_factor->get_values();
    auto u_values = u_factor->get_values();

    for (auto r = 0; r < l_factor->get_size()[0]; r++) {
        for (auto c = 0; c < l_factor->get_size()[1]; c++) {
            l_values[l_factor->get_size()[0] * r + c] = 0.0;
        }
    }

    for (auto r = 0; r < u_factor->get_size()[0]; r++) {
        for (auto c = 0; c < u_factor->get_size()[1]; c++) {
            u_values[u_factor->get_size()[0] * r + c] = 0.0;
        }
    }

    for (auto i = 0; i < mtx->get_size()[0]; i++) {
        ValueType pivot = mtx_values[mtx->get_size()[0] * i + i];
        if (abs(pivot) < PIVOT_THRESHOLD) {
            pivot += PIVOT_AUGMENTATION;
        }

        // Stores l_values in row-major format.
        l_values[l_factor->get_size()[0] * i + i] = 1.0;
        for (auto j = i + 1; j < mtx->get_size()[0]; j++) {
            l_values[l_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i] / pivot;
        }

        // Stores u_values in col-major format.
        u_values[u_factor->get_size()[0] * i + i] = pivot;
        for (auto j = i + 1; j < mtx->get_size()[1]; j++) {
            u_values[u_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i];
        }
    }
}


// Computes the dense LU factors of the diagonal blocks of submatrix_11.
template <typename ValueType, typename IndexType>
void factorize_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks, const IndexType* partitions,
    IndexType* a_cur_row_ptrs,
    const std::shared_ptr<matrix::Csr<ValueType, IndexType>> matrices,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> l_factors,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> u_factors)
{
    // using dense = matrix::Dense<ValueType>;
    // const auto stride = 1;
    // for (auto block = 0; block < num_blocks; block++) {
    //     const auto block_length
    //         = static_cast<size_type>(partitions[block + 1] -
    //                                  partitions[block]);
    //     const dim<2> block_size = {block_length, block_length};

    //     auto a_submtx = share(dense::create(exec));
    //     as<ConvertibleTo<dense>>(a_linop[block])
    //         ->convert_to(a_submtx.get());

    //     auto l_submtx = share(dense::create(exec));
    //     as<ConvertibleTo<dense>>(l_factors[block])
    //         ->convert_to(l_submtx.get());

    //     auto u_submtx = share(dense::create(exec));
    //     as<ConvertibleTo<dense>>(u_factors[block])
    //         ->convert_to(u_submtx.get());

    //     factorize_kernel(a_submtx.get(), l_submtx.get(), u_submtx.get());
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);

// Step 3 of computing LU factors of submatrix_12. Sets up the
// nonzero entries of submatrix_12 of U factor.
template <typename ValueType, typename IndexType>
void factorize_off_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, IndexType split_index,
    IndexType num_blocks, const IndexType* partitions,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> a_off_diagonal_blocks,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> triang_factors,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> off_diagonal_blocks)
{
    // using dense = matrix::Dense<ValueType>;
    // size_type stride = 1;
    // array<ValueType> res_values =
    //     array<ValueType>(exec, factor_01.get_num_elems());
    // exec->copy(factor_01.get_num_elems(), factor_01->get_values(),
    //            res_values.get_data());
    // for (auto block = 0; block < num_blocks; block++) {
    //     auto nnz_in_block =
    //         factor_01->row_ptrs[block + 1] - factor_01->row_ptrs[block];
    //     if (nnz_in_block > 0) {
    //         const auto num_rows = partitions[block + 1] - partitions[block];
    //         const auto num_cols = factor_01.get_num_elems() / num_rows;
    //         dim<2> dim_u = {num_rows, num_cols};
    //         auto residuals =
    //             dense::create(exec, dim_u,
    //                           array<ValueType>::view(
    //                               exec, factor_01.get_num_elems(),
    //                               res_values),
    //                           stride);

    //         // Solves l_factor_00 * X = u_factor_01
    //         const auto row_index_begin = partitions[block];
    //         const auto row_index_end = partitions[block + 1];
    //         const auto block_length = static_cast<size_type>(
    //             partitions[block + 1] - partitions[block]);
    //         const dim<2> dim_l = {block_length, block_length};
    //         auto factor_00_submtx = share(dense::create(exec));
    //         as<ConvertibleTo<dense>>(factor_00[block])
    //             ->convert_to(factor_00_submtx.get());
    //         auto l_values = factor_00_submtx->get_values();
    //         auto u_values = &factor_01->get_values()[row_index_begin];
    //         auto rhs =
    //             dense::create(exec, dim_u,
    //                           array<ValueType>::view(
    //                               exec, factor_01.get_num_elems(), u_values),
    //                           stride);
    //         lower_triangular_solve_kernel(dim_l, l_values, dim_u, u_values);

    //         // Computes residual vectors.
    //         dim<2> dim_rnorm = {1, dim_u[1]};
    //         array<ValueType> rnorms_values = {exec, dim_rnorm[1]};
    //         auto rnorms = dense::create(exec, dim_rnorm, rnorms_values,
    //         stride); rnorms.fill(0.0); auto solution = dense::create(
    //             exec, dim_u,
    //             array<ValueType>::view(exec, dim_u[0] * dim_u[1], u_values),
    //             stride);
    //         exec->copy(factor_01.get_num_elems(), rhs, residuals);
    //         auto one =
    //             gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec);
    //         auto minus_one =
    //             gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec);
    //         factor_00_submtx->apply(solution, minus_one, residuals, one);

    //         // Computes residual norms.
    //         residuals.get()->compute_norm2(rnorms.get());
    //         for (auto i = 0; i < rnorms->get_size()[1]; ++i) {
    //             if (std::abs(rnorms->get_values()[i]) > 1e-8) {
    //                 break;
    //             } else {
    //                 // Refine if required.
    //             }
    //         }
    //     }
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void spdgemm_in(std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
                IndexType split_index, IndexType block_size,
                IndexType block_index, ValueType alpha,
                const IndexType* partition_idxs, const LinOp* linop_left,
                const LinOp* linop_right, LinOp* linop_result)
{
    using dense = matrix::Dense<ValueType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    auto left_mtx = share(csr::create(exec));
    as<ConvertibleTo<csr>>(linop_left[block_index])->convert_to(left_mtx.get());
    auto right_mtx = share(csr::create(exec));
    as<ConvertibleTo<csr>>(linop_right[block_index])
        ->convert_to(right_mtx.get());
    const auto values_l = left_mtx->get_values();
    const auto row_idxs_l = left_mtx->get_col_idxs();
    const auto col_ptrs_l = left_mtx->get_row_ptrs();
    const auto values_r = right_mtx->get_values();
    const auto col_idxs_r = right_mtx->get_col_idxs();
    const auto row_ptrs_r = right_mtx->get_row_ptrs();
    const auto ptr_span_r = {row_ptrs_r[partition_idxs[block_index]],
                             row_ptrs_r[partition_idxs[block_index + 1]]};
    const auto ptr_span_l = {col_ptrs_l[partition_idxs[block_index]],
                             col_ptrs_l[partition_idxs[block_index + 1]]};
    const auto num_cols_r = (ptr_span_r.end - ptr_span_r.begin) /
                            static_cast<IndexType>(block_size);
    const auto num_rows_l = (ptr_span_l.end - ptr_span_l.begin) /
                            static_cast<IndexType>(block_size);
    auto schur_complement = share(dense::create(exec));
    as<ConvertibleTo<csr>>(linop_result[0])->convert_to(schur_complement.get());
    auto schur_complement_values = schur_complement->get_values();
    for (auto i = 0; i < block_size; i++) {
        for (auto j = 0; j < num_rows_l; j++) {
            for (auto k = 0; k < num_cols_r; k++) {
                auto col_index_l = ptr_span_l + num_rows_l * i + j;
                auto value_l = values_l[col_index_l];
                auto row = row_idxs_l[col_index_l];

                auto row_index_r = ptr_span_r + num_cols_r * i + k;
                auto value_r = values_r[row_index_r];
                auto col = col_idxs_r[row_index_r];

                schur_complement_values[size[1] * row + col] +=
                    (alpha * value_l * value_r);
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void spdgemm(std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks,
             const IndexType* partition_idxs, const LinOp* linop_left,
             const LinOp* linop_right, LinOp* schur_complement)
{
    for (IndexType block = 0; block < num_blocks; block++) {
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        ValueType coeff = -1.0;
        spdgemm_in(exec, block_size, block, coeff, partition_idxs, linop_left,
                   linop_right, schur_complement);
    }
}

// Computes the schur complement of submatrix_22.
template <typename ValueType, typename IndexType>
void compute_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks,
    const IndexType* partitions,
    const std::shared_ptr<matrix::Csr<ValueType, IndexType>> l_factors_10,
    const std::shared_ptr<matrix::Csr<ValueType, IndexType>> u_factors_01,
    std::shared_ptr<LinOp> schur_complement_in)
{
    // using csr = matrix::Csr<ValueType, IndexType>;
    // const auto l_factor = as<csr>(l_factors_10);
    // const auto u_factor = as<csr>(u_factors_01);
    // const auto schur_complement = as<csr>(schur_complement_in);
    // spdgemm(exec, num_blocks, partitions, l_factor, u_factor,
    // schur_complement);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL);

template <typename ValueType, typename IndexType>
void factorize_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks, const IndexType* partitions,
    IndexType* a_cur_row_ptrs,
    const std::shared_ptr<matrix::Dense<ValueType>> matrices,
    std::shared_ptr<matrix::Dense<ValueType>> l_factors,
    std::shared_ptr<matrix::Dense<ValueType>> u_factors)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_SCHUR_COMPLEMENT_KERNEL);

template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Arrow<ValueType, IndexType>> create_factor(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Arrow<ValueType, IndexType>* a_)
{
    // using dense = matrix::Dense<ValueType>;
    // using csr = matrix::Csr<ValueType, IndexType>;
    std::unique_ptr<matrix::Arrow<ValueType, IndexType>> l_factor = {exec};
    // size_type stride = 1;
    // const auto partitions = a_matrix->get_partition_idxs();
    // const auto num_blocks = a_matrix->get_num_elems() - 1;
    // const auto size = a_matrix->get_submatrix_00()->get_size();
    // const auto max_col = size[0] + size[1] + 1;
    // const auto split_index = partitions[num_blocks];
    // const auto a_row_ptrs = a_matrix->get_const_row_ptrs();
    // const auto a_col_idxs = a_matrix->get_const_col_idxs();
    // const auto a_values = a_matrix->get_const_values();
    // array<IndexType> tmp1_array = {exec, size[0] + 1};
    // array<IndexType> tmp2_array = {exec, size[0] + 1};
    // array<IndexType> tmp3_array = {exec, size[0] + 1};
    // tmp1_array.fill(0);
    // tmp2_array.fill(0);
    // tmp3_array.fill(0);
    // auto a_cur_row_ptrs = tmp1_array.get_data();
    // auto l_cur_row_ptrs = tmp2_array.get_data();
    // auto u_cur_row_ptrs = tmp3_array.get_data();

    // // Creates submatrix_00 of l_factor.
    // auto factor_submtx_00 = factor->get_submatrix_00();
    // for (auto block = 0; block < num_blocks; block++) {
    //     if (1) {
    //         const dim<2> block_size = {partitions[block + 1] -
    //         partitions[block],
    //                                    partitions[block + 1] -
    //                                    partitions[block]};
    //         auto tmp = array<ValueType>(exec, block_size[0] * block_size[1]);
    //         tmp.fill(0.0);
    //         factor_submtx_00->push_back(std::move(
    //             dense::create(exec, block_size, std::move(tmp), stride)));
    //     }
    // }

    // // Creates submatrix_11 of l_factor.
    // {
    //     auto l_submtx_11 = l_factor->get_submatrix_11();
    //     const dim<2> block_size = {size[0] -
    //     static_cast<size_type>(partitions[num_blocks]),
    //                                size[1] -
    //                                static_cast<size_type>(partitions[num_blocks])};
    //     auto tmp = array<ValueType>(exec, block_size[0] * block_size[1]);
    //     tmp.fill(0.0);
    //     l_submtx_11 = dense::create(exec, block_size, std::move(tmp),
    //     stride);
    // }

    // // Computes nnz of submatrix_01 of l_factor.
    // IndexType nnz_l = 0;
    // auto factor_submtx_10 = factor->get_submatrix_01();
    // exec->copy(size[0] + 1, a_row_ptrs, a_cur_row_ptrs);
    // for (auto block = 0; block < num_blocks; block++) {
    //     IndexType col_min = 0;
    //     IndexType row_min = 0;
    //     IndexType num_occurences = 0;
    //     gko::span row_span{partitions[block], partitions[block + 1]};
    //     const auto block_length = row_span.end() - row_span.begin();
    //     auto remaining_nnz = compute_remaining_nnz_row_check(row_span,
    //         a_row_ptrs, a_cur_row_ptrs);
    //     auto factor_values = factor_submtx_10[block].get_values();
    //     auto factor_col_idxs = factor_submtx_10[block].get_col_idxs();
    //     while (remaining_nnz > 0) {
    //         find_min_col(size, row_span, &col_min, a_col_idxs, a_row_ptrs,
    //         a_cur_row_ptrs); push_wavefront(size, row_span, col_min,
    //         &remaining_nnz,
    //                        a_col_idxs, a_row_ptrs,
    //                        &nnz_l, factor_cur_row_ptrs);
    //     }
    // }

    // // Creates submatrix_10 of factor.
    // components::prefix_sum(exec, l_cur_row_ptrs, size[0]);
    // array<IndexType> factor_col_idxs = {exec, nnz_l};
    // array<ValueType> factor_values = {exec, nnz_l};
    // u_col_idxs.fill(0);
    // u_values.fill(0.0);
    // exec->copy(size[0] + 1, a_row_ptrs, a_cur_row_ptrs);
    // exec->copy(size[0] + 1, l_row_ptrs, l_cur_row_ptrs);
    // for (auto block = 0; block < num_blocks; block++) {
    //     gko::span row_span = {partition_idxs[block], partition_idxs[block +
    //     1]} const auto block_length = partition_idxs[block + 1] -
    //     partition_idxs[block]; IndexType col_min = 0; IndexType remaining_nnz
    //     = 0; auto remaining_nnz = compute_remaining_nnz_row_check(
    //         row_span, a_row_ptrs, a_cur_row_ptrs);
    //     while (remaining_nnz > 0) {
    //         find_min_col(size, row_span, &col_min, a_col_idxs, a_row_ptrs,
    //         a_cur_row_ptrs); copy_off_diagonal_block(split_index, row_span,
    //         col_min,
    //             &remaining_nnz,
    //             a_values, a_col_idxs, tmp_a_row_ptrs,
    //             u_values, u_col_idxs, u_row_ptrs);
    //     }
    // }
    // submtx_01 = matrix::Csr<ValueType, IndexType>::create(exec, size,
    // std::move(u_values),
    //     std::move(u_col_idxs), std::move(u_row_ptrs));

    return l_factor;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Arrow<ValueType, IndexType>> create_l_factor(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Arrow<ValueType, IndexType>* a_matrix)
{
    std::unique_ptr<matrix::Arrow<ValueType, IndexType>> l_factor;
    create_factor(exec, a_matrix->get_submatrix_00(),
                  a_matrix->get_submatrix_10(), a_matrix->get_submatrix_11(),
                  l_factor->get_submatrix_00(), l_factor->get_submatrix_10(),
                  l_factor->get_submatrix_11());
    return l_factor;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Arrow<ValueType, IndexType>> create_u_factor(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Arrow<ValueType, IndexType>* a_matrix)
{
    std::unique_ptr<matrix::Arrow<ValueType, IndexType>> u_factor;
    create_factor(exec, a_matrix->get_submatrix_00(),
                  a_matrix->get_submatrix_01(), a_matrix->get_submatrix_11(),
                  u_factor->get_submatrix_00(), u_factor->get_submatrix_01(),
                  u_factor->get_submatrix_11());
    return u_factor;
}

}  // namespace arrow_lu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
