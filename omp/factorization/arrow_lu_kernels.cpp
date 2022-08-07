#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"


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
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_lu_workspace<ValueType, IndexType>* workspace,
    const gko::matrix::Csr<ValueType, IndexType>* mtx)
{
    auto submtx_11 = workspace->get_submatrix_11();
    auto submtx_12 = workspace->get_submatrix_12();
    auto submtx_21 = workspace->get_submatrix_21();
    auto submtx_22 = workspace->get_submatrix_22();
    auto partitions = workspace->get_partitions();

    array<IndexType> row_ptrs_src_cur_array = {exec, submtx_12->size[0] + 1};
    array<IndexType> row_ptrs_dst_cur_array = {exec, submtx_12->size[0] + 1};
    array<IndexType> col_ptrs_dst_cur_array = {exec, submtx_12->size[0] + 1};

    initialize_submatrix_11(exec, partitions, mtx, submtx_11);
    factorize_submatrix_11(exec, partitions, mtx, submtx_11);

    // preprocess_submatrix_12(exec, partitions, mtx, submtx_12,
    // row_ptrs_src_cur_array, row_ptrs_dst_cur_array);
    // initialize_submatrix_12(exec, partitions, mtx, submtx_12,
    // row_ptrs_src_cur_array); factorize_submatrix_12(exec, partitions,
    // submtx_11, submtx_12);

    // preprocess_submatrix_21(exec, partitions, mtx, submtx_21,
    // col_ptrs_dst_cur_array, row_ptrs_src_cur_array);
    // initialize_submatrix_21(exec, workspace->get_partitions(), mtx,
    // workspace->get_submatrix_21()); factorize_submatrix_21(exec,
    // workspace->get_partitions(), workspace->get_submatrix_11(),
    // workspace->get_submatrix_21());

    // initialize_submatrix_22(exec, workspace->get_partitions(),
    //                        workspace->get_submatrix_11(),
    //                        workspace->get_submatrix_12(),
    //                        workspace->get_submatrix_21(),
    //                        workspace->get_submatrix_22());
    // factorize_submatrix_22(exec, workspace->get_submatrix_22());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);


// Checks rows [row_start, ..., row_end] of submatrix_12 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <typename IndexType>
void find_min_col(const IndexType* row_ptrs_mtx, const IndexType* col_idxs_mtx,
                  IndexType* row_ptrs_cur, dim<2> size, IndexType row_start,
                  IndexType row_end, IndexType* col_min_out,
                  IndexType* row_min_out, IndexType* num_occurences_out)
{
    const auto max_row = static_cast<IndexType>(size[0] + size[1]);
    auto col_min = max_row;
    auto row_min = max_row;
    IndexType num_occurences = 0;
    for (auto row = row_start; row < row_end; row++) {
        auto row_index = row_ptrs_cur[row];
        auto col = col_idxs_mtx[row_index];
        num_occurences =
            ((col == col_min) && (row_index < row_ptrs_mtx[row + 1]))
                ? (num_occurences + 1)
                : num_occurences;
        row_min = ((col < col_min) && (row_index < row_ptrs_mtx[row + 1]))
                      ? row
                      : row_min;
        col_min = ((col < col_min) && (row_index < row_ptrs_mtx[row + 1]))
                      ? col
                      : col_min;
    }
    *col_min_out = col_min;
    *row_min_out = row_min;
    *num_occurences_out = num_occurences;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_FIND_MIN_COL_KERNEL);


// Checks if there is a nonzero entry in rows [row_start, ..., row_end] of input
// matrix. Performs the checks by comparing row_ptrs[row] with row_ptrs_cur[row]
// for row = row_start, ..., row_end. Returns the number of remaining nzs.
template <typename IndexType>
IndexType compute_remaining_nnz_row_check(const IndexType* row_ptrs_mtx,
                                          IndexType* row_ptrs_cur,
                                          IndexType row_start,
                                          IndexType row_end)
{
    auto remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        remaining_nnz += ((row_ptrs_mtx[row + 1] > row_ptrs_cur[row])
                              ? row_ptrs_mtx[row + 1] - row_ptrs_cur[row]
                              : 0);
    }
    return remaining_nnz;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_REMAINING_NNZ_ROW_CHECK_KERNEL);


// Checks if there is a nonzero entry in rows row_start, ..., row_end of input
// matrix. Performs the checks by comparing column indices. Returns the number
// of remaining nzs.
template <typename IndexType>
IndexType compute_remaining_nnz_col_check(const IndexType* col_idxs_mtx,
                                          IndexType* row_ptrs_cur,
                                          IndexType row_start,
                                          IndexType row_end, IndexType col_end)
{
    IndexType remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        auto row_index = row_ptrs_cur[row];
        while ((col_idxs_mtx[row_index] < col_end) &&
               (row_index < row_ptrs_cur[row + 1])) {
            row_ptrs_cur[row] += 1;
            row_index = row_ptrs_cur[row];
            remaining_nnz += 1;
        }
    }
    return remaining_nnz;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_REMAINING_NNZ_COL_CHECK_KERNEL);


// Initializes the dense diagonal blocks of submatrix_11.
template <typename ValueType, typename IndexType>
void initialize_submatrix_11(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11)
{
    using dense = matrix::Dense<ValueType>;
    const size_type stride = 1;
    const auto partition_idxs = partitions->get_const_data();
    const auto num_blocks = submtx_11->num_blocks;
    auto l_factors = submtx_11->l_factors.begin();

    for (auto block = 0; block < num_blocks; block++) {
        const dim<2> block_size = {
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block]),
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block])};
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        submtx_11->l_factors.push_back(std::move(
            dense::create(exec, block_size, std::move(tmp_array), stride)));
    }

    auto u_factors = submtx_11->l_factors.begin();
    for (auto block = 0; block < num_blocks; block++) {
        const dim<2> block_size = {
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block]),
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block])};
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        submtx_11->u_factors.push_back(std::move(
            dense::create(exec, block_size, std::move(tmp_array), stride)));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INITIALIZE_SUBMATRIX_11_KERNEL);


// Step 2 for computing LU factors of submatrix_11. Computes the dense
// LU factors of the diagonal blocks of submatrix_11.
template <typename ValueType, typename IndexType>
void factorize_submatrix_11(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11)
{
    using dense = matrix::Dense<ValueType>;
    const auto num_blocks = submtx_11->num_blocks;
    const auto split_index = submtx_11->split_index;
    const auto stride = 1;
    const auto partition_idxs = partitions->get_const_data();
    const auto values_mtx = mtx->get_const_values();
    const auto col_idxs_mtx = mtx->get_const_col_idxs();
    const auto row_ptrs_mtx = mtx->get_const_row_ptrs();
    auto row_ptrs = submtx_11->row_ptrs_cur.get_data();
    size_type nnz_l_factor = 0;
    size_type nnz_u_factor = 0;
    exec->copy(split_index + 1, row_ptrs_mtx, row_ptrs);
    IndexType nnz_l = 0;
    IndexType nnz_u = 0;
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        const auto len = static_cast<size_type>(partition_idxs[block + 1] -
                                                partition_idxs[block]);
        const dim<2> block_size = {len, len};
        const auto num_elems_dense = static_cast<size_type>(len * len);
        nnz_l += (len * len + len) / 2;
        nnz_u += (len * len + len) / 2;

        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        auto system_mtx =
            dense::create(exec, block_size, std::move(tmp_array), stride);
        convert_csr_2_dense<ValueType, IndexType>(
            block_size, row_ptrs, col_idxs_mtx, values_mtx, system_mtx.get(),
            row_start, row_end);
        submtx_11->diag_blocks.push_back(std::move(system_mtx));

        // factorize_kernel(submtx_11->dense_diagonal_blocks[block].get(),
        //                 submtx_11->dense_l_factors[block].get(),
        //                 submtx_11->dense_u_factors[block].get());
    }
    submtx_11->nnz_l = nnz_l;
    submtx_11->nnz_u = nnz_u;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL);


/// Step 1 of computing LU factors of submatrix_12. Computes the number of
// nonzero entries of submatrix_12.
template <typename ValueType, typename IndexType>
void preprocess_submatrix_12(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12,
    array<IndexType>& row_ptrs_cur_src_array,
    array<IndexType>& row_ptrs_cur_dst_array)
{
    const auto split_index = submtx_12->split_index;
    const auto max_col = submtx_12->size[0] + submtx_12->size[1] + 1;
    const auto num_blocks = submtx_12->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto row_ptrs_mtx = mtx->get_const_row_ptrs();
    const auto col_idxs_mtx = mtx->get_const_col_idxs();
    auto row_ptrs_cur = submtx_12->row_ptrs_cur.get_data();
    auto block_row_ptrs = submtx_12->block_ptrs.get_data();
    auto nnz_per_block = submtx_12->nnz_per_block.get_data();
    exec->copy(submtx_12->size[0], row_ptrs_mtx, row_ptrs_cur);
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        const auto len = partition_idxs[block + 1] - partition_idxs[block];
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        auto remaining_nnz = compute_remaining_nnz_row_check(
            row_ptrs_mtx, row_ptrs_cur, row_start, row_end);
        auto nnz_count = 0;
        IndexType col_min = 0;
        IndexType row_min = 0;
        IndexType num_occurences = 0;
        nnz_per_block[block] = remaining_nnz;
        while (remaining_nnz > 0) {
            find_min_col(row_ptrs_mtx, col_idxs_mtx, row_ptrs_cur,
                         submtx_12->size, row_start, row_end, &col_min,
                         &row_min, &num_occurences);
            // Finds the rows with col = col_min and updates remaining_nnz.
            bool found_nonzero_column = false;
            for (auto row = row_start; row < row_end; row++) {
                const auto row_index = row_ptrs_cur[row];
                const auto col = col_idxs_mtx[row_index];
                if ((col == col_min) &&
                    (row_ptrs_cur[row] < row_ptrs_mtx[row + 1])) {
                    row_ptrs_cur[row] += 1;
                    remaining_nnz -= 1;
                    found_nonzero_column = true;
                }
            }
            // Updates nnz_count and sets new col_min to be equal to
            // the previously maximum column.
            nnz_count = (found_nonzero_column && (col_min >= split_index))
                            ? nnz_count + len
                            : nnz_count;
            col_min = max_col;
        }
        block_row_ptrs[block + 1] = nnz_count;
    }
    components::prefix_sum(exec, block_row_ptrs, num_blocks + 1);
    submtx_12->nnz = block_row_ptrs[num_blocks + 1];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PREPROCESS_SUBMATRIX_12_KERNEL);


// Step 2 of computing LU factors of submatrix_12. Initializes
// the nonzero entries of submatrix_12.
template <typename ValueType, typename IndexType>
void initialize_submatrix_12(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12,
    array<IndexType>& row_ptrs_cur_src_array)
{
    const auto num_blocks = submtx_12->num_blocks;
    const auto split_index = submtx_12->split_index;
    {
        array<IndexType> row_ptrs_tmp = {exec, submtx_12->size[0] + 1};
        array<IndexType> col_idxs_tmp = {
            exec, static_cast<size_type>(submtx_12->nnz)};
        array<ValueType> values_tmp = {exec,
                                       static_cast<size_type>(submtx_12->nnz)};
        submtx_12->mtx = share(matrix::Csr<ValueType, IndexType>::create(
            exec, submtx_12->size, std::move(values_tmp),
            std::move(col_idxs_tmp), std::move(row_ptrs_tmp)));
    }
    const auto partition_idxs = partitions->get_const_data();
    const auto row_ptrs_mtx = mtx->get_const_row_ptrs();
    const auto values_mtx = mtx->get_const_values();
    const auto col_idxs_mtx = mtx->get_const_col_idxs();
    auto row_ptrs_cur = submtx_12->row_ptrs_cur.get_data();
    auto values = submtx_12->mtx->get_values();
    auto col_idxs = submtx_12->mtx->get_col_idxs();
    auto row_ptrs = submtx_12->mtx->get_row_ptrs();
    exec->copy(submtx_12->size[0], row_ptrs_mtx, row_ptrs_cur);
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        IndexType num_rhs = 0;
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        IndexType num_occurences = 0;
        while (1) {
            IndexType col_min = 0;
            IndexType row_min = 0;
            IndexType remaining_nnz = 0;
            find_min_col(row_ptrs_mtx, col_idxs_mtx, row_ptrs_cur,
                         submtx_12->size, row_start, row_end, &col_min,
                         &row_min, &num_occurences);
            remaining_nnz = compute_remaining_nnz_row_check(
                row_ptrs_mtx, row_ptrs_cur, row_start, row_end);
            if (remaining_nnz == 0) {
                break;
            }
            auto row_index_submtx_12 = row_ptrs[row_min];
            for (auto row = row_start; row < row_end; row++) {
                const auto row_index_mtx = row_ptrs_cur[row];
                const auto col = col_idxs_mtx[row_index_mtx];
                const auto row_index_submtx_12 = row_ptrs[row];
                values[row_index_submtx_12] =
                    (col == col_min) ? values_mtx[row_index_mtx] : 0.0;
                col_idxs[row_index_submtx_12] = col_min - split_index;
                row_ptrs[row] += 1;
                if (col == col_min) {
                    row_ptrs_cur[row] += 1;
                }
            }
            num_rhs += 1;
        }
    }

    // Resets row_ptrs to original position.
#pragma omp parallel for schedule(dynamic)
    for (auto block = num_blocks - 1; block >= 0; block--) {
        const auto row_end = partition_idxs[block + 1];
        const auto row_start = partition_idxs[block];
        for (auto row = row_end; row >= row_start; row--) {
            row_ptrs[row] = (row > 0) ? row_ptrs[row - 1] : 0;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INITIALIZE_SUBMATRIX_12_KERNEL);


// Step 3 of computing LU factors of submatrix_12. Sets up the
// nonzero entries of submatrix_12 of U factor.
template <typename ValueType, typename IndexType>
void factorize_submatrix_12(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12)
{
    using dense = matrix::Dense<ValueType>;
    const auto stride = 1;
    const auto num_blocks = submtx_12->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto block_row_ptrs = submtx_12->block_ptrs.get_const_data();
    array<ValueType> residuals = array<ValueType>(exec, submtx_12->nnz);
    exec->copy(submtx_12->nnz, submtx_12->mtx->get_values(),
               residuals.get_data());
    //#pragma omp parallel for schedule(dynamic)
    //    for (auto block = 0; block < num_blocks; block++) {
    //        if (submtx_12->nz_per_block.get_data()[block] > 0) {
    //            const auto block_size =
    //                static_cast<size_type>(partition_idxs[block + 1] -
    //                partition_idxs[block]);
    //            const dim<2> dim_tmp = {static_cast<size_type>(block_size),
    //                                    static_cast<size_type>(block_size)};
    //            const dim<2> dim_rhs = {block_size,
    //                                    static_cast<size_type>(block_row_ptrs[block
    //                                    + 1] - block_row_ptrs[block]) /
    //                                    block_size}
    //
    //            auto l_factor_tmp = submtx_11->dense_l_factors[block].get();
    //            auto values_l_factor = l_factor_tmp->get_values();
    //            auto values_12 =
    //                &submtx_12->mtx->get_values()[block_row_ptrs[block]];
    //
    //            lower_triangular_solve_kernel(dim_tmp, values_l_factor,
    //            dim_rhs,
    //                                          values_12);
    //
    //            auto num_elems = dim_rhs[0] * dim_rhs[1];
    //            auto values_residual =
    //                &residuals.get_data()[block_row_ptrs[block]];
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
    //                                  exec, dim_rhs[0] * dim_rhs[1],
    //                                  values_12),
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
    //                if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
    //                    std::cout << "i: " << i << "abs values: "
    //                              << std::abs(residual_norm->get_values()[i])
    //                              << ", block_index: " << block << '\n';
    //                    break;
    //                }
    //            }
    //        }
    //    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL);


// Step 1 of computing LU factors of submatrix_21. Computes the number of
// nonzeros of submatrix_21 of L factor.
template <typename ValueType, typename IndexType>
void preprocess_submatrix_21(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21,
    array<IndexType>& col_ptrs_dst_array, array<IndexType>& row_ptrs_dst_array)
{
    const auto split_index = submtx_21->split_index;
    const auto num_blocks = submtx_21->num_blocks;
    const auto size = submtx_21->size;
    const auto col_idxs_mtx = mtx->get_const_col_idxs();
    const auto row_ptrs_mtx = mtx->get_const_row_ptrs();
    const auto values_mtx = mtx->get_const_values();
    const auto partition_idxs = partitions->get_const_data();
    auto row_ptrs_cur_mtx = submtx_21->row_ptrs_cur.get_data();
    auto block_col_ptrs_submtx_21 = submtx_21->block_ptrs.get_data();
    auto nnz_per_block_submtx_21 = submtx_21->nnz_per_block.get_data();
    auto col_ptrs_submtx_21 = submtx_21->col_ptrs_cur.get_data();
    // Compressed representation storage.
    std::vector<IndexType> storage_rows;
    std::vector<IndexType> storage_row_ptrs;
    //    array<IndexType> storage_block_row_ptrs_array = {exec,
    //    static_cast<IndexType>(num_blocks) + 1}; auto storage_block_row_ptrs =
    //    storage_block_row_ptrs_array.get_data();
    //    // For handling compressed representation storage.
    //    IndexType num_elems_mtx = 0;
    //    IndexType max_num_elems_mtx = size[0] + size[1];
    //    // Initialize storage and submtx_21, arrays that are required.
    //    storage_rows.resize(size[0] + size[1]);
    //    storage_row_ptrs.resize(size[0] + size[1]);
    //    storage_block_row_ptrs_array.fill(0);
    //    submtx_21->block_ptrs.fill(0);
    //    submtx_21->col_ptrs_cur.fill(0);
    //    exec->copy(size[0], &row_ptrs_mtx[split_index], row_ptrs_cur_mtx);
    //    // Main loop.
    //    auto nz_count_total_submtx_21 = 0;
    //    for (auto block = 0; block < num_blocks; block++) {
    //        IndexType nz_count_submtx_21 = 0;
    //        const auto col_start = partition_idxs[block];
    //        const auto col_end = partition_idxs[block + 1];
    //        const auto block_size =
    //            partition_idxs[block + 1] - partition_idxs[block];
    //#pragma omp parallel for schedule(dynamic) shared(col_start, col_end,
    // block_size) reduction(+:nz_count_submtx_21)
    //        for (auto row = 0; row < size[0]; row++) {
    //            auto row_index_mtx = row_ptrs_cur_mtx[row];
    //            auto col_mtx = col_idxs_mtx[row_index_mtx];
    //            // If current (row, col_mtx) entry remains in the current
    //            // partition, increment nz_count_submtx_21 and update
    //            // col_ptrs_submtx_21.
    //            if ((col_mtx >= col_start) && (col_mtx < col_end)) {
    //                nz_count_submtx_21 += block_size;
    //#pragma omp critical
    //                {
    //                    storage_block_row_ptrs[block + 1] += 1;
    //                }
    //
    //                // If stored entries exceed size.
    //                if (num_elems_mtx + 1 >= max_num_elems_mtx) {
    //                    max_num_elems_mtx += (size[0] + size[1]);
    //                    storage_rows.resize(max_num_elems_mtx);
    //                    storage_row_ptrs.resize(max_num_elems_mtx);
    //                }
    //
    //                // Inserts entry.
    //                storage_rows[num_elems_mtx] = row + split_index;
    //                storage_row_ptrs[num_elems_mtx] = row_ptrs_cur_mtx[row];
    //                num_elems_mtx += 1;
    //                row_ptrs_cur_mtx[row] += 1;
    //                row_index_mtx = row_ptrs_cur_mtx[row];
    //                col_mtx = col_idxs_mtx[row_index_mtx];
    //
    //                // Increment col_ptrs_submtx_21.
    //                for (auto col_submtx_21 = col_start; col_submtx_21 <
    //                col_end;
    //                     col_submtx_21++) {
    //                    col_ptrs_submtx_21[col_submtx_21 + 1] += 1;
    //                }
    //
    //                // Increments row_ptrs_cur_mtx[row] until it reaches the
    //                // beginning of the next block.
    //                while ((col_mtx >= col_start) && (col_mtx < col_end)) {
    //                    row_ptrs_cur_mtx[row] += 1;
    //                    row_index_mtx = row_ptrs_cur_mtx[row];
    //                    col_mtx = col_idxs_mtx[row_index_mtx];
    //
    //                    if (num_elems_mtx + 1 >= max_num_elems_mtx) {
    //                        max_num_elems_mtx += (size[0] + size[1]);
    //                        storage_rows.resize(max_num_elems_mtx);
    //                        storage_row_ptrs.resize(max_num_elems_mtx);
    //                    }
    //
    //                    // Stores (row, row_ptr).
    //                    storage_rows[num_elems_mtx] = row + split_index;
    //                    storage_row_ptrs[num_elems_mtx] =
    //                    row_ptrs_cur_mtx[row]; num_elems_mtx += 1;
    //                }
    //            }
    //        }
    //        // Updates nonzero information for submtx_21->
    //        nz_count_total_submtx_21 += nz_count_submtx_21;
    //        nz_per_block_submtx_21[block] = nz_count_submtx_21;
    //        block_col_ptrs_submtx_21[block + 1] = nz_count_total_submtx_21;
    //    }
    //    submtx_21->nnz = nz_count_total_submtx_21;
    //
    //    components::prefix_sum(exec, block_row_ptrs, num_blocks + 1);
    //
    //    //{
    //    //    array<IndexType> rows_in = {
    //    //        exec, static_cast<size_type>(num_elems_mtx)};
    //    //    array<IndexType> row_ptrs_in = {
    //    //        exec, static_cast<size_type>(num_elems_mtx)};
    //    //    array<IndexType> block_ptrs_in = {
    //    //        exec, static_cast<size_type>(num_blocks) + 1};
    //    //    for (auto i = 0; i < num_elems_mtx; i++) {
    //    //        rows_in.get_data()[i] = compressed_rows_mtx[i];
    //    //    }
    //    //    for (auto i = 0; i < num_elems_mtx; i++) {
    //    //        row_ptrs_in.get_data()[i] = compressed_row_ptrs_mtx[i];
    //    //    }
    //    //    for (auto i = 0; i < num_blocks + 1; i++) {
    //    //        block_ptrs_in.get_data()[i] =
    //    //            compressed_block_row_ptrs_mtx.get_data()[i];
    //    //    }
    //
    //    //    submtx_21->block_storage =
    //    // std::make_shared<gko::factorization::block_csr_storage<IndexType>>(
    //    //            rows_in, row_ptrs_in, block_ptrs_in);
    //    //}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PREPROCESS_SUBMATRIX_21_KERNEL);


// Step 2 of computing LU factors of submatrix_21. Sets up the
// nonzeros of submatrix_21 of U factor.
template <typename ValueType, typename IndexType>
void initialize_submatrix_21(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21)
{
    const auto size = submtx_21->size;
    const auto split_index = submtx_21->split_index;
    const auto num_blocks = submtx_21->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto values_mtx = mtx->get_const_values();
    const auto row_ptrs_mtx = mtx->get_const_row_ptrs();
    const auto col_idxs_mtx = mtx->get_const_col_idxs();
    auto row_ptrs_cur_mtx = submtx_21->row_ptrs_cur.get_data();
    auto values_submtx_21 = submtx_21->mtx->get_values();
    auto col_ptrs_submtx_21 = submtx_21->mtx->get_row_ptrs();
    auto row_idxs_submtx_21 = submtx_21->mtx->get_col_idxs();
    auto col_ptrs_cur_submtx_21 = submtx_21->col_ptrs_cur.get_data();
    auto block_col_ptrs_submtx_21 = submtx_21->block_ptrs.get_data();
    auto storage = submtx_21->block_storage;
    auto block_row_ptrs = storage->block_ptrs.get_data();
    auto rows = storage->rows.get_data();
    auto row_ptrs_local = storage->row_ptrs.get_data();

    // Initializes row_ptrs_cur_mtx to (2, 1)-submatrix of
    // mtx row_ptrs.
    exec->copy(size[0] + 1, &row_ptrs_mtx[split_index], row_ptrs_cur_mtx);

// Updates col_ptrs[partition_idxs[block]] += block_col_ptrs[block] for each
// 0 <= block < num_blocks.
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto col_idx = partition_idxs[block];
        col_ptrs_submtx_21[col_idx] = block_col_ptrs_submtx_21[block];
    }
    // Performs a reduction on col_ptrs of partition. @Replace with ginkgo
    // partial sum function.
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        const auto row_start = block_row_ptrs[block];
        const auto row_end = block_row_ptrs[block + 1];
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        const auto num_cols = (submtx_21->block_ptrs.get_data()[block + 1] -
                               submtx_21->block_ptrs.get_data()[block]) /
                              block_size;
        for (auto col = partition_idxs[block] + 1;
             col < partition_idxs[block + 1]; col++) {
            col_ptrs_submtx_21[col] = col_ptrs_submtx_21[col - 1] + num_cols;
        }
    }

// Main loop.
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        const auto col_start = partition_idxs[block];
        const auto col_end = partition_idxs[block + 1];
        IndexType block_size = 0;
        IndexType num_cols = 0;

        IndexType row_start = 0;
        IndexType row_end = 0;

#pragma omp critical
        {
            block_size = partition_idxs[block + 1] - partition_idxs[block];
            row_start = block_row_ptrs[block];
            row_end = block_row_ptrs[block + 1];
            num_cols = (row_end - row_start) / block_size;
        }
        IndexType col_min = partition_idxs[block];
        IndexType row_min = split_index;

        // Sets row_idxs_submtx_21 for entries in submatrix block.
        exec->copy(block_size, &col_ptrs_submtx_21[col_start],
                   &col_ptrs_cur_submtx_21[col_start]);
        for (auto row_index = row_start; row_index < row_end; row_index++) {
            auto row_mtx = rows[row_index];
            for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
                 col_submtx_21++) {
                auto col_index = col_ptrs_cur_submtx_21[col_submtx_21];
                row_idxs_submtx_21[col_index] = row_mtx;
                col_ptrs_cur_submtx_21[col_submtx_21] += 1;
            }
        }

        // Copies numerical values from mtx to arrow_submatrix_21.mtx.
        // Modifies row_ptrs (of arrow_submatrix_21) while copying.
        exec->copy(block_size, &col_ptrs_submtx_21[col_start],
                   &col_ptrs_cur_submtx_21[col_start]);
        for (auto row_index = row_start; row_index < row_end; row_index++) {
            auto row_index_mtx = row_ptrs_local[row_index];
            for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
                 col_submtx_21++) {
                const auto col_index = col_ptrs_cur_submtx_21[col_submtx_21];
                const auto col_mtx = col_idxs_mtx[row_index_mtx];
                values_submtx_21[col_index] =
                    ((col_submtx_21 == col_mtx) && (col_mtx < col_end))
                        ? values_mtx[row_index_mtx]
                        : 0.0;
                row_ptrs_local[row_index] += 1;
                row_index_mtx = row_ptrs_local[row_index];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INITIALIZE_SUBMATRIX_21_KERNEL);


// Step 3 of computing submatrix_21 of L factor. Sets up the
// nonzeros of submatrix_21 of L factor.
template <typename ValueType, typename IndexType>
void factorize_submatrix_21(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11,
    factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21)
{
    using dense = matrix::Dense<ValueType>;
    const auto num_blocks = submtx_21->num_blocks;
    const auto block_col_ptrs = submtx_21->block_ptrs.get_const_data();
    const auto partition_idxs = partitions->get_const_data();

    auto nnz_per_block = submtx_21->nnz_per_block.get_data();
    auto residuals = array<ValueType>(exec, submtx_21->nnz);
    exec->copy(submtx_21->nnz, submtx_21->mtx->get_values(),
               residuals.get_data());
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        if (nnz_per_block[block] > 0) {
            // Sets up dimensions of local data.
            const auto stride = 1;
            const IndexType row_idx = partition_idxs[block];
            const IndexType block_size = static_cast<size_type>(
                partition_idxs[block + 1] - partition_idxs[block]);
            const dim<2> dim_tmp = {static_cast<size_type>(block_size),
                                    static_cast<size_type>(block_size)};

            // Computes the left solution to a local linear system.
            const dim<2> dim_rhs = {
                static_cast<size_type>(block_col_ptrs[block + 1] -
                                       block_col_ptrs[block]) /
                    block_size,
                block_size};
            auto system_mtx = share(dense::create(exec));
            as<ConvertibleTo<dense>>(submtx_11->u_factors[block].get())
                ->convert_to(system_mtx.get());
            const auto values_u_factor = system_mtx.get()->get_values();
            auto values_21 =
                &submtx_21->mtx->get_values()[block_col_ptrs[block]];
            upper_triangular_left_solve_kernel(dim_tmp, values_u_factor,
                                               dim_rhs, values_21);

            // Computes residual vectors.
            size_type one = 1;
            const dim<2> dim_rnorm = {one, dim_rhs[1]};
            const auto num_elems = dim_rhs[0] * dim_rhs[1];
            auto values_residual = &residuals.get_data()[block_col_ptrs[block]];
            auto residual_vectors = dense::create(
                exec, dim_rhs,
                array<ValueType>::view(exec, num_elems, values_residual),
                stride);
            array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
            values_rnorm.fill(0.0);
            auto residual_norm =
                dense::create(exec, dim_rnorm, values_rnorm, stride);

            // Matrix is stored in CSC format here so either transpose it or use
            // it as row vector x matrix operations
            auto u_factor = share(dense::create(
                exec, dim_tmp,
                array<ValueType>::view(exec, dim_tmp[0] * dim_tmp[1],
                                       values_u_factor),
                stride));
            auto solution =
                dense::create(exec, dim_rhs,
                              array<ValueType>::view(
                                  exec, dim_rhs[0] * dim_rhs[1], values_21),
                              stride);

            // MM multiplication (check if format is ok)
            auto x = solution->get_values();
            auto b = residual_vectors->get_values();
            auto u_vals = u_factor->get_values();
            for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
                for (auto col_l = 0; col_l < dim_tmp[1]; col_l++) {
                    for (auto intern_index = 0; intern_index < dim_rhs[1];
                         intern_index++) {
                        b[row_b + dim_rhs[0] * col_l] -=
                            u_vals[intern_index + dim_tmp[0] * col_l] *
                            x[row_b + dim_rhs[0] * intern_index];
                    }
                }
            }

            // Compute residual norms.
            auto r = residual_vectors.get();
            r->compute_norm2(residual_norm.get());
            for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
                if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
                    break;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL);


// Computes the schur complement of submatrix_22.
template <typename ValueType, typename IndexType>
void initialize_submatrix_22(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11,
    const factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12,
    const factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21,
    factorization::arrow_submatrix_22<ValueType, IndexType>* submtx_22)
{
    using dense = matrix::Dense<ValueType>;
    auto schur_complement = share(dense::create(exec));
    as<ConvertibleTo<dense>>(submtx_22->mtx.get())
        ->convert_to(schur_complement.get());
    const dim<2> size = {submtx_21->size[0], submtx_12->size[1]};
    const auto num_blocks = submtx_11->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
#pragma omp parallel for schedule(dynamic)
    for (IndexType block = 0; block < num_blocks; block++) {
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        // Synchronization here is required for writting (updating the entries).
        ValueType coeff = -1.0;
        // spdgemm_blocks(exec, size, block_size, submtx_11, submtx_12,
        //               submtx_21, submtx_22, block, coeff);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INITIALIZE_SUBMATRIX_22_KERNEL);


template <typename ValueType, typename IndexType>
void spdgemm_blocks(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType block_size,
    const factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11,
    const factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12,
    const factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21,
    matrix::Dense<ValueType>* schur_complement, IndexType block_index,
    ValueType alpha)
{
    using dense = matrix::Dense<ValueType>;
    const auto split_index = submtx_12->split_index;
    const auto block_col_ptrs = submtx_21->block_ptrs.get_const_data();
    const auto block_row_ptrs = submtx_12->block_ptrs.get_const_data();
    const auto num_rows_21 =
        (block_col_ptrs[block_index + 1] - block_col_ptrs[block_index]) /
        block_size;
    const auto num_cols_12 =
        (block_row_ptrs[block_index + 1] - block_row_ptrs[block_index]) /
        block_size;
    const auto values_21 = submtx_21->mtx->get_values();
    const auto row_idxs_21 = submtx_21->mtx->get_col_idxs();
    const auto values_12 = submtx_12->mtx->get_values();
    const auto col_idxs_12 = submtx_21->mtx->get_col_idxs();
    auto schur_complement_values = schur_complement->get_values();
    for (auto i = 0; i < block_size; i++) {
        for (auto j = 0; j < num_rows_21; j++) {
            for (auto k = 0; k < num_cols_12; k++) {
                auto col_index_21 =
                    block_col_ptrs[block_index] + num_rows_21 * i + j;
                auto value_21 = values_21[col_index_21];
                auto row = row_idxs_21[col_index_21] - split_index;

                auto row_index_12 =
                    block_row_ptrs[block_index] + num_cols_12 * i + k;
                auto value_12 = values_12[row_index_12];
                auto col = col_idxs_12[row_index_12];

#pragma omp critical
                {
                    schur_complement_values[size[1] * row + col] +=
                        (alpha * value_21 * value_12);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL);


// Computes L and U factors of submatrix_22.
template <typename ValueType, typename IndexType>
void factorize_submatrix_22(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_submatrix_22<ValueType, IndexType>* submtx_22)
{
    using dense = matrix::Dense<ValueType>;
    const auto system_mtx = share(dense::create(exec));
    as<ConvertibleTo<dense>>(submtx_22->mtx.get())
        ->convert_to(system_mtx.get());

    auto l_factor = share(dense::create(exec));
    as<ConvertibleTo<dense>>(submtx_22->l_factor.get())
        ->convert_to(l_factor.get());

    auto u_factor = share(dense::create(exec));
    as<ConvertibleTo<dense>>(submtx_22->u_factor.get())
        ->convert_to(u_factor.get());

    factorize_kernel(system_mtx.get(), l_factor.get(), u_factor.get());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL);


// Solves triangular system. L-factor is stored in row-major ordering.
template <typename ValueType>
void lower_triangular_solve_kernel(dim<2> dim_l_factor,
                                   const ValueType* l_factor, dim<2> dim_rhs,
                                   ValueType* rhs_matrix)
{
    // Computes rhs_matrix[dim_rhs[1]*row + num_rhs] - dot_product[num_rhs]
    //  = l_factor[row*dim_l_factor[1] + col]*rhs_matrix[col * dim_rhs[1] +
    //  num_rhs]
    // for all rows and col = 0, ..., row-1
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
        auto pivot = l_factor[dim_l_factor[0] * row + row];
        for (auto num_rhs = 0; num_rhs < dim_rhs[1]; num_rhs++) {
            rhs_matrix[dim_rhs[1] * row + num_rhs] =
                rhs_matrix[dim_rhs[1] * row + num_rhs] / pivot;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL);

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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL);

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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_LEFT_SOLVE_KERNEL);

// Converts spare matrix in CSR format to dense.
template <typename ValueType, typename IndexType>
void convert_csr_2_dense(dim<2> size, const IndexType* row_ptrs,
                         const IndexType* col_idxs, const ValueType* values,
                         matrix::Dense<ValueType>* dense_mtx,
                         const IndexType col_start, const IndexType col_end)
{
    auto values_mtx = dense_mtx->get_values();
    auto num_rows = dense_mtx->get_size()[0];
    for (auto row_local = 0; row_local < size[0]; row_local++) {
        IndexType col_old = -1;
        auto row = row_local + col_start;
        auto row_index = row_ptrs[row];
        auto col_cur = col_idxs[row_index];
        while ((col_cur < col_end) && (col_old < col_cur)) {
            auto col_local = col_cur - col_start;
            values_mtx[num_rows * row_local + col_local] = values[row_index];
            col_old = col_cur;
            row_index += 1;
            col_cur = col_idxs[row_index];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL);


// Factorize kernels.
template <typename ValueType>
void factorize_kernel(const matrix::Dense<ValueType>* mtx,
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

        // store in row-major format
        l_values[l_factor->get_size()[0] * i + i] = 1.0;
        for (auto j = i + 1; j < mtx->get_size()[0]; j++) {
            l_values[l_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i] / pivot;
        }

        // store in col-major format
        u_values[u_factor->get_size()[0] * i + i] = pivot;
        for (auto j = i + 1; j < mtx->get_size()[1]; j++) {
            u_values[u_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FACTORIZE_KERNEL);


}  // namespace arrow_lu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
