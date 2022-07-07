#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>

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


// Checks rows row_start, ..., row_end of submatrix_12 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <typename ValueType, typename IndexType>
void find_min_col(
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    const IndexType* row_ptrs, const IndexType* col_idxs, IndexType row_start,
    IndexType row_end, IndexType* col_min_out, IndexType* row_min_out,
    IndexType* num_occurences_out)
{
    IndexType max_row = submtx_12.size[0] + submtx_12.size[1] + 1;
    IndexType num_occurences = 0;
    auto col_min = max_row;
    auto row_min = max_row;
    auto row_ptrs_current = submtx_12.row_ptrs_tmp.get_data();
    for (auto row = row_start; row < row_end; row++) {
        auto row_index = row_ptrs_current[row];
        auto col = col_idxs[row_index];
        num_occurences = ((col == col_min) && (row_index < row_ptrs[row + 1]))
                             ? (num_occurences + 1)
                             : num_occurences;
        row_min = ((col < col_min) && (row_index < row_ptrs[row + 1]))
                      ? row
                      : row_min;
        col_min = ((col < col_min) && (row_index < row_ptrs[row + 1]))
                      ? col
                      : col_min;
    }
    *col_min_out = col_min;
    *row_min_out = row_min;
    *num_occurences_out = num_occurences;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FIND_MIN_COL_12_KERNEL);

// Checks rows row_start, ..., row_end of submatrix_21 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <typename ValueType, typename IndexType>
void find_min_col(
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    const IndexType* row_ptrs, const IndexType* col_idxs, IndexType row_start,
    IndexType row_end, IndexType* col_min_out, IndexType* row_min_out,
    IndexType* num_occurences_out)
{
    IndexType max_row = submtx_21.size[0] + submtx_21.size[1] + 1;
    auto col_min = max_row;
    auto row_min = max_row;
    IndexType num_occurences = 0;
    auto row_ptrs_current = submtx_21.row_ptrs_tmp.get_data();
    for (auto i = row_start; i < row_end; ++i) {
        auto row_index = row_ptrs_current[i - row_start];
        auto col = col_idxs[row_index];
        num_occurences = ((col == col_min) && (row_index < row_ptrs[i + 1]))
                             ? (num_occurences + 1)
                             : num_occurences;
        row_min =
            ((col < col_min) && (row_index < row_ptrs[i + 1])) ? i : row_min;
        col_min =
            ((col < col_min) && (row_index < row_ptrs[i + 1])) ? col : col_min;
    }
    *col_min_out = col_min;
    *row_min_out = row_min;
    *num_occurences_out = num_occurences;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FIND_MIN_COL_21_KERNEL);

// Checks if there is a nonzero entry in rows row_start, ..., row_end of input
// matrix. Performs the checks by comparing row_ptrs[row] with
// row_ptrs_current[row] for row = row_start, ..., row_end.
template <typename IndexType>
IndexType symbolic_count_row_check(IndexType* row_ptrs_current,
                                   const IndexType* row_ptrs_source,
                                   const IndexType row_start,
                                   const IndexType row_end)
{
    auto remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        remaining_nnz += ((row_ptrs_source[row + 1] > row_ptrs_current[row])
                              ? row_ptrs_source[row + 1] - row_ptrs_current[row]
                              : 0);
    }
    return remaining_nnz;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SYMBOLIC_COUNT_ROW_CHECK_KERNEL);

// Checks if there is a nonzero entry in rows row_start, ..., row_end of input
// matrix. Performs the checks by comparing column indices.
template <typename IndexType>
IndexType symbolic_count_col_check(IndexType* row_ptrs_current,
                                   const IndexType* col_idxs_source,
                                   const IndexType row_start,
                                   const IndexType row_end,
                                   const IndexType col_end)
{
    IndexType remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        auto row_index = row_ptrs_current[row];
        while ((col_idxs_source[row_index] < col_end) &&
               (row_index < row_ptrs_current[row + 1])) {
            row_ptrs_current[row] += 1;
            row_index = row_ptrs_current[row];
            remaining_nnz += 1;
        }
    }
    return remaining_nnz;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SYMBOLIC_COUNT_COL_CHECK_KERNEL);

// Computes LU factorization of submatrix_11.
template <typename ValueType, typename IndexType>
void factorize_submatrix_11(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_partitions<IndexType>& partitions)
{
    step_1_impl_assemble(global_mtx, submtx_11, partitions);
    step_2_impl_factorize(global_mtx, submtx_11, partitions);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL);

// Updates entries of LU factorization of submatrix_12.
template <typename ValueType, typename IndexType>
void factorize_submatrix_12(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_partitions<IndexType>& partitions,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12)
{
    auto nnzs = static_cast<size_type>(submtx_12.nz);
    auto exec = submtx_12.exec;
    auto size = submtx_12.size;

    // Symbolic count and setup or row_ptrs.
    step_1_impl_symbolic_count(global_mtx, submtx_12, partitions);

    // Copies nonzero block-sized columns from blocks of (1, 2)-submatrix
    // of global_mtx to submtx_12.mtx in CSR format.
    step_2_impl_assemble(global_mtx, submtx_12, partitions);

    // Applies triangular solves on arrow_submtx_21 using the diagonal
    // blocks, stored in submtx_11. After the application of triangular solves,
    // submtx_12.mtx contains the submatrix-(1, 2) of the U factor.
    step_3_impl_factorize(submtx_11, submtx_12, partitions);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL);

// Updates entries of LU factorization of submatrix_21.
template <typename ValueType, typename IndexType>
void factorize_submatrix_21(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_partitions<IndexType>& partitions,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21)
{
    step_1_impl_symbolic_count(global_mtx, submtx_21, partitions);

    // Allocates memory and assembles values of arrow_submatrix_21.
    step_2_impl_assemble(global_mtx, submtx_21, partitions);

    // applies triangular solves on arrow_submatrix_21 using the diagonal
    // blocks, stored in arrow_submatrix_11.
    step_3_impl_factorize(submtx_11, submtx_21, partitions);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL);

template <typename ValueType, typename IndexType>
void factorize_submatrix_22(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_partitions<IndexType>& partitions,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_submatrix_22<ValueType, IndexType>& submtx_22)
{
    step_1_impl_compute_schur_complement(submtx_11, submtx_12, submtx_21,
                                         submtx_22, partitions);
    step_2_impl_factorize(submtx_22);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL);

template <typename ValueType, typename IndexType>
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    gko::matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_lu_workspace<ValueType, IndexType>* workspace)
{
    factorize_submatrix_11(global_mtx, workspace->mtx_.submtx_11_,
                           workspace->mtx_.partitions_);
    factorize_submatrix_12(global_mtx, workspace->mtx_.partitions_,
                           workspace->mtx_.submtx_11_,
                           workspace->mtx_.submtx_12_);
    // factorize_submatrix_21(global_mtx, workspace->mtx_.partitions_,
    //     workspace->mtx_.submtx_11_, workspace->mtx_.submtx_21_);
    // factorize_submatrix_22(global_mtx, workspace->mtx_.partitions_,
    //     workspace->mtx_.submtx_11_, workspace->mtx_.submtx_12_,
    //     workspace->mtx_.submtx_21_, workspace->mtx_.submtx_22_);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);


// Step 1 for computing LU factors of submatrix_11. Initializes the dense
// diagonal blocks of submatrix_11.
template <typename ValueType, typename IndexType>
void step_1_impl_assemble(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto exec = submtx_11.exec;
    size_type stride = 1;
    auto partition_idxs = partitions.data.get_data();
    using dense = matrix::Dense<ValueType>;
    auto num_blocks = submtx_11.num_blocks;
    array<array<ValueType>> l_factor_values_array =
        array<array<ValueType>>(exec, num_blocks);
    array<array<ValueType>> u_factor_values_array =
        array<array<ValueType>>(exec, num_blocks);

#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto len = static_cast<size_type>(partition_idxs[block + 1] -
                                          partition_idxs[block]);
        const dim<2> block_size = {len, len};
        auto values_array = l_factor_values_array.get_data()[block];
        values_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        values_array.fill(0);
        submtx_11.dense_l_factors.get_data()[block] =
            dense::create(exec, block_size, std::move(values_array), stride);
    }

#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto len = static_cast<size_type>(partition_idxs[block + 1] -
                                          partition_idxs[block]);
        const dim<2> block_size = {len, len};
        auto values_array = u_factor_values_array.get_data()[block];
        values_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        values_array.fill(0);
        submtx_11.dense_u_factors.get_data()[block] =
            dense::create(exec, block_size, std::move(values_array), stride);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_1_IMPL_ASSEMBLE_11_KERNEL);

// Step 2 for computing LU factors of submatrix_11. Computes the dense
// LU factors of the diagonal blocks of submatrix_11.
template <typename ValueType, typename IndexType>
void step_2_impl_factorize(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_partitions<IndexType>& partitions)
{
    using dense = matrix::Dense<ValueType>;
    auto exec = submtx_11.exec;
    const auto partition_idxs = partitions.get_const_data();
    auto row_ptrs = submtx_11.row_ptrs_tmp.get_data();
    const auto col_idxs = global_mtx->get_const_col_idxs();
    const auto values = global_mtx->get_const_values();
    const auto stride = 1;
    IndexType nnz_l_factor = 0;
    IndexType nnz_u_factor = 0;
    exec->copy(submtx_11.split_index + 1, global_mtx->get_const_row_ptrs(),
               row_ptrs);
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < submtx_11.num_blocks; block++) {
        const auto len = static_cast<size_type>(partition_idxs[block + 1] -
                                                partition_idxs[block]);
        const auto num_elems_dense = static_cast<size_type>(len * len);
        dim<2> block_size = {len, len};
        nnz_l_factor += (len * len + len) / 2;
        nnz_u_factor += (len * len + len) / 2;
        array<ValueType> values_tmp = {exec, num_elems_dense};
        values_tmp.fill(0.0);
        submtx_11.dense_diagonal_blocks.get_data()[block] =
            dense::create(exec, block_size, std::move(values_tmp), stride);

        auto dense_l_factor = submtx_11.dense_l_factors.get_data()[block].get();
        auto dense_u_factor = submtx_11.dense_u_factors.get_data()[block].get();
        auto dense_block =
            submtx_11.dense_diagonal_blocks.get_data()[block].get();
        auto row_start = partition_idxs[block];
        auto row_end = partition_idxs[block + 1];
        convert_csr_2_dense<ValueType, IndexType>(block_size, row_ptrs,
                                                  col_idxs, values, dense_block,
                                                  row_start, row_end);
        factorize_kernel(dense_block, dense_l_factor, dense_u_factor);
    }
    submtx_11.nnz_l_factor = nnz_l_factor;
    submtx_11.nnz_u_factor = nnz_u_factor;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_11_KERNEL);

// Step 1 of computing LU factors of submatrix_12. Computes the number of
// nonzero entries of submatrix_12.
template <typename ValueType, typename IndexType>
void step_1_impl_symbolic_count(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto exec = submtx_12.exec;
    auto size = submtx_12.size;
    auto partition_idxs = partitions.get_data();
    const auto row_ptrs_src = global_mtx->get_const_row_ptrs();
    const auto col_idxs_src = global_mtx->get_const_col_idxs();
    auto row_ptrs_current = submtx_12.row_ptrs_tmp.get_data();
    auto block_row_ptrs = submtx_12.block_row_ptrs.get_data();
    auto nz_per_block = submtx_12.nz_per_block.get_data();
    exec->copy(submtx_12.size[0], row_ptrs_src, row_ptrs_current);
    IndexType nnz_submatrix_12_count =
        0;  // Number of nonzeros in submtx_12.mtx
    IndexType col_min = 0;
    IndexType row_min = 0;
    IndexType num_occurences = 0;
    IndexType max_col = submtx_12.size[0] + submtx_12.size[1] + 1;
    auto num_blocks = submtx_12.num_blocks;
    auto split_index = submtx_12.split_index;
    submtx_12.row_ptrs_tmp2 = {exec, size[0] + 1};
    submtx_12.row_ptrs_tmp2.fill(0);
    auto row_ptrs_submtx_12_current = submtx_12.row_ptrs_tmp2.get_data();
    // #pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        // Computes the number of remaining nonzeros in current block of
        // submatrix (1, 2) of global_mtx.
        auto remaining_nnz = symbolic_count_row_check(
            row_ptrs_current, row_ptrs_src, row_start, row_end);
        nz_per_block[block] = remaining_nnz;
        // Computes the number of nonzeros of the column-sparse block and
        // sets up the block_row, row_ptr indices for the current block of
        // submatrix_12.
        while (remaining_nnz > 0) {
            // Finds the (row_min. col_min) entry in submatrix_12 that has the
            // smallest column in the current block.
            find_min_col(submtx_12, row_ptrs_src, col_idxs_src, row_start,
                         row_end, &col_min, &row_min, &num_occurences);

            // Increments row_ptrs_current[row] for all entries (row, col) s.t.
            // row in (row_start, row_end) and col =
            // col_idxs[row_ptrs_current[row]].
            bool found_nonzero_column = false;
            for (auto row = row_start; row < row_end; row++) {
                auto r_ptr = row_ptrs_current[row];
                auto col = col_idxs_src[r_ptr];
                if ((col == col_min) &&
                    (row_ptrs_current[row] < row_ptrs_src[row + 1])) {
                    row_ptrs_current[row] += 1;
                    remaining_nnz -= 1;
                    found_nonzero_column = true;
                    // Updates number row_ptrs of submatrix_12.
                    row_ptrs_submtx_12_current[row] += 1;
                }
            }

            // Updates nonzero count, num_cols and col_min
            nnz_submatrix_12_count =
                (found_nonzero_column && (col_min >= split_index))
                    ? nnz_submatrix_12_count + block_size
                    : nnz_submatrix_12_count;
            col_min = max_col;
        }

        // Refreshes the row_ptrs_current
        auto num_elems = row_end - row_start;
        exec->copy(num_elems, &row_ptrs_src[row_start],
                   &row_ptrs_current[row_start]);
        // Updates the block_row_ptrs array.
        block_row_ptrs[block + 1] =
            nnz_submatrix_12_count;  // this should be done here but there
                                     // should be something in the description
                                     // of this function to detail it.
    }

    // Computes row_ptrs for submtx_12.mtx.
    for (auto block = 0; block < num_blocks; block++) {
        auto row_start = partition_idxs[block];
        auto row_end = partition_idxs[block + 1];
        row_ptrs_submtx_12_current[row_start + 1] += block_row_ptrs[block];
        for (auto row = row_start + 2; row < row_end; row++) {
            row_ptrs_submtx_12_current[row] +=
                row_ptrs_submtx_12_current[row - 1];
        }
    }
    // Updates the total nnz count of submatrix_12.
    submtx_12.nz = nnz_submatrix_12_count;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_12_KERNEL);

// Step 2 of computing LU factors of submatrix_12. Initializes
// the nonzero entries of submatrix_12.
template <typename ValueType, typename IndexType>
void step_2_impl_assemble(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto exec = submtx_12.exec;
    auto nnzs = submtx_12.nz;
    array<IndexType> col_idxs_tmp = {exec, nnzs};
    col_idxs_tmp.fill(0);
    array<ValueType> values_tmp = {exec, nnzs};
    values_tmp.fill(0);
    auto row_ptrs_current = submtx_12.row_ptrs_tmp.get_data();
    auto partition_idxs = partitions.get_data();
    auto values_source = global_mtx->get_const_values();
    auto col_idxs_source = global_mtx->get_const_col_idxs();
    auto row_ptrs_source = global_mtx->get_const_row_ptrs();
    exec->copy(submtx_12.size[0], row_ptrs_source, row_ptrs_current);
    auto values = values_tmp.get_data();
    auto col_idxs = col_idxs_tmp.get_data();
    auto row_ptrs = submtx_12.row_ptrs_tmp2.get_data();
    auto num_blocks = submtx_12.num_blocks;
    auto split_index = submtx_12.split_index;
    std::vector<IndexType> row_ptrs_local;


    // #pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto block_size = partition_idxs[block + 1] - partition_idxs[block];
        auto row_start = partition_idxs[block];
        auto row_end = partition_idxs[block + 1];
        auto col_start = partition_idxs[block];
        auto col_end = partition_idxs[block + 1];
        IndexType num_occurences = 0;
        // While there exists nonzero entries in block of submatrix_12.
        while (1) {
            IndexType col_min = 0;
            IndexType row_min = 0;
            IndexType remaining_nnz = 0;

            // Find (row_min, col_min) s.t. col_min is the minimum column
            // in the wavefront.
            find_min_col(submtx_12, row_ptrs_source, col_idxs_source, col_start,
                         col_end, &col_min, &row_min, &num_occurences);
            // Update remaining_nnz for entries in block of global_mtx.
            remaining_nnz =
                symbolic_count_row_check((IndexType*)row_ptrs_current,
                                         row_ptrs_source, row_start, row_end);
            if (remaining_nnz == 0) {
                break;
            }
            // Copies all nonzero entries in the wavefront of (1, 2) subblock
            // iu global_mtx to submatrix_12.mtx.
            for (auto row = row_start; row < row_end; row++) {
                auto row_index = row_ptrs_current[row];
                auto col = col_idxs_source[row_index];
                // Copies value and column. Updates row_ptr.
                auto row_index_output = row_ptrs[row];
                values[row_index_output] =
                    (col == col_min) ? values_source[row_index] : 0.0;
                col_idxs[row_index_output] = col_min - split_index;
                row_ptrs[row] += 1;
                // Increment wavefront ptr.
                if (col == col_min) {
                    row_ptrs_current[row] += 1;
                }
            }
        }
    }

    // Resets row_ptrs to original position.
    for (auto i = submtx_12.num_blocks - 1; i >= 0; i--) {
        auto row_end = partition_idxs[i + 1];
        auto row_start = partition_idxs[i];
        for (auto j = row_end; j >= row_start; j--) {
            row_ptrs[j] = (j > 0) ? row_ptrs[j - 1] : 0;
        }
    }
    // Creates submtx_12.mtx.
    submtx_12.mtx = share(matrix::Csr<ValueType, IndexType>::create(
        exec, submtx_12.size, std::move(values_tmp), std::move(col_idxs_tmp),
        std::move(submtx_12.row_ptrs_tmp2)));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_12_KERNEL);

// Step 3 of computing LU factors of submatrix_12. Sets up the
// nonzero entries of submatrix_12 of U factor.
template <typename ValueType, typename IndexType>
void step_3_impl_factorize(
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto exec = submtx_11.exec;
    auto block_row_ptrs_data = submtx_12.block_row_ptrs.get_data();
    auto partition_idxs = partitions.get_data();
    auto dense_l_factors = submtx_11.dense_l_factors;
    auto stride = 1;
    auto num_blocks = submtx_12.num_blocks;
    auto nz_per_block = submtx_12.nz_per_block.get_data();
    using dense = matrix::Dense<ValueType>;
    array<ValueType> residuals = array<ValueType>(exec, submtx_12.nz);
    exec->copy(submtx_12.nz, submtx_12.mtx->get_values(), residuals.get_data());
#pragma omp parallel for schedule(dynamic)
    for (IndexType block = 0; block < num_blocks; block++) {
        if (nz_per_block[block] > 0) {
            auto block_size = static_cast<size_type>(partition_idxs[block + 1] -
                                                     partition_idxs[block]);
            dim<2> dim_tmp = {static_cast<size_type>(block_size),
                              static_cast<size_type>(block_size)};
            dim<2> dim_rhs;
            dim_rhs[0] = static_cast<size_type>(block_size);
            dim_rhs[1] = static_cast<size_type>(
                (block_row_ptrs_data[block + 1] - block_row_ptrs_data[block]) /
                block_size);

            // debug
            std::cout << "block: " << block << '\n';
            std::cout << "dim_rhs[1]: " << dim_rhs[1] << '\n';
            std::cout << "partition_idxs[block+1]: "
                      << partition_idxs[block + 1] << '\n';

            // Extracts values from dense block of submtx_11 and rhs in
            // submtx_12 in CSR format.
            auto values_l_factor =
                submtx_11.dense_l_factors.get_data()[block].get()->get_values();
            auto values_12 =
                &submtx_12.mtx->get_values()[block_row_ptrs_data[block]];
            lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                          values_12);

            // auto num_elems = dim_rhs[0] * dim_rhs[1];
            // auto values_residual =
            //     &residuals.get_data()[block_row_ptrs_data[block]];
            // auto residual_vectors = dense::create(
            //     exec, dim_rhs,
            //     array<ValueType>::view(exec, num_elems, values_residual),
            //     stride);

            // dim<2> dim_rnorm = {1, dim_rhs[1]};
            // array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
            // values_rnorm.fill(0.0);
            // auto residual_norm = dense::create(exec, dim_rnorm,
            //     values_rnorm, stride);

            // auto l_factor = share(dense::create(
            //     exec, dim_tmp,
            //     array<ValueType>::view(exec, dim_tmp[0] * dim_tmp[1],
            //                             values_l_factor), stride));

            // auto solution =
            //     dense::create(exec, dim_rhs,
            //                     array<ValueType>::view(
            //                         exec, dim_rhs[0] * dim_rhs[1],
            //                         values_12),
            //                     stride);

            // // Performs MM multiplication.
            // auto x = solution->get_values();
            // auto b = residual_vectors->get_values();
            // auto l_vals = l_factor->get_values();
            // for (auto row_l = 0; row_l < dim_tmp[0]; row_l++) {
            //     for (auto col_b = 0; col_b < dim_rhs[1]; col_b++) {
            //         for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
            //             b[dim_rhs[1]*row_l + col_b] -=
            //             l_vals[dim_tmp[1]*row_l + row_b]*x[dim_rhs[1]*row_b +
            //             col_b];
            //         }
            //     }
            // }

            // // Computes residual norms.
            // auto r = residual_vectors.get();
            // r->compute_norm2(residual_norm.get());
            // for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
            //     if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
            //         std::cout << "i: " << i << "abs values: " <<
            //         std::abs(residual_norm->get_values()[i]) << ",
            //         block_index: " << block << '\n'; break;
            //     }
            // }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_3_IMPL_FACTORIZE_12_KERNEL);

// Step 1 of computing LU factors of submatrix_21. Computes the number of
// nonzeros of submatrix_21 of L factor.
template <typename ValueType, typename IndexType>
void step_1_impl_symbolic_count(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto exec = submtx_21.exec;
    auto size = submtx_21.size;
    auto split_index = submtx_21.split_index;
    auto num_blocks = submtx_21.num_blocks;
    auto nz_count_total = 0;
    auto partition_idxs = partitions.get_data();
    const auto col_idxs_source = global_mtx->get_const_col_idxs();
    const auto row_ptrs_source = global_mtx->get_const_row_ptrs();
    const auto values_source = global_mtx->get_const_values();
    auto row_ptrs_source_current = submtx_21.row_ptrs_tmp.get_data();
    auto block_col_ptrs_submtx_21 = submtx_21.block_col_ptrs.get_data();
    auto nz_per_block_submtx_21 = submtx_21.nz_per_block.get_data();
    submtx_21.col_ptrs_tmp.fill(0);
    auto col_ptrs_submtx_21 = submtx_21.col_ptrs_tmp.get_data();
    std::vector<IndexType> row_ptrs_local;
    row_ptrs_local.resize(size[0] + size[1]);
    std::vector<IndexType> row_list_local;
    row_list_local.resize(size[0] + size[1]);
    IndexType num_entries_in_row_ptrs_local = 0;
    IndexType max_num_entries = size[0] + size[1];
    submtx_21.row_list_local_ptrs = {exec, num_blocks + 1};

    exec->copy(size[0], &row_ptrs_source[split_index], row_ptrs_source_current);
    // Main loop.
    for (auto block = 0; block < num_blocks; block++) {
        IndexType nz_count = 0;
        const auto col_start = partition_idxs[block];
        const auto col_end = partition_idxs[block + 1];
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
#pragma omp parallel for schedule(dynamic) shared(col_start, col_end, block_size) reduction(+:nz_count)
        for (auto row = 0; row < size[0]; row++) {
            auto row_index_source = row_ptrs_source_current[row];
            auto col_source = col_idxs_source[row_index_source];
            // If current (row, col_source) entry remains in the current
            // partition, increment nz_count and update col_ptrs_submtx_21.
            if ((col_source >= col_start) && (col_source < col_end)) {
                nz_count += block_size;

                // If stored entries exceed size.
                if (num_entries_in_row_ptrs_local + 1 >= max_num_entries) {
                    max_num_entries += (size[0] + size[1]);
                    row_list_local.resize(max_num_entries);
                    row_ptrs_local.resize(max_num_entries);
                }

                // Stores.
                row_list_local[num_entries_in_row_ptrs_local] = row;
                row_ptrs_local[num_entries_in_row_ptrs_local] =
                    row_ptrs_source_current[row];
                num_entries_in_row_ptrs_local += 1;

                // Increment col_ptrs_submtx_21.
                for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
                     col_submtx_21++) {
                    col_ptrs_submtx_21[col_submtx_21 + 1] += 1;
                }

                // Increments row_ptrs_source_current[row] until it reaches the
                // beginning of the next block.
                while ((col_source >= col_start) && (col_source < col_end)) {
                    row_ptrs_source_current[row] += 1;
                    row_index_source = row_ptrs_source_current[row];
                    col_source = col_idxs_source[row_index_source];
                }
            }
        }
        // Updates nonzero information for submtx_21.
        nz_count_total += nz_count;
        block_col_ptrs_submtx_21[block + 1] = nz_count_total;
        nz_per_block_submtx_21[block] = nz_count;
        submtx_21.row_list_local_ptrs.get_data()[block + 1] =
            nz_count_total / block_size;
    }
    submtx_21.nz = nz_count_total;
    // Copies row_list_local and row_ptrs_local to gko::array.
    submtx_21.row_list_local = {exec, num_entries_in_row_ptrs_local};
    submtx_21.row_ptrs_local = {exec, num_entries_in_row_ptrs_local};
    exec->copy(num_entries_in_row_ptrs_local, &row_list_local[0],
               submtx_21.row_list_local.get_data());
    exec->copy(num_entries_in_row_ptrs_local, &row_ptrs_local[0],
               submtx_21.row_ptrs_local.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_21_KERNEL);

// Step 2 of computing LU factors of submatrix_21. Sets up the
// nonzeros of submatrix_21 of U factor.
template <typename ValueType, typename IndexType>
void step_2_impl_assemble(
    const matrix::Csr<ValueType, IndexType>* global_mtx,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto nz_submtx_21 = static_cast<size_type>(submtx_21.nz);
    auto exec = submtx_21.exec;
    auto size = submtx_21.size;
    const auto row_start = submtx_21.split_index;
    const IndexType row_end = (IndexType)global_mtx->get_size()[0];
    auto split_index = submtx_21.split_index;
    auto num_blocks = submtx_21.num_blocks;

    // Creates submtx_21.mtx
    {
        auto col_ptrs_tmp = array<IndexType>(exec, size[1] + 1);
        col_ptrs_tmp.fill(0);
        auto row_idxs_tmp = array<IndexType>(exec, nz_submtx_21);
        row_idxs_tmp.fill(0);
        auto values_tmp = array<ValueType>(exec, nz_submtx_21);
        values_tmp.fill(0.0);
        submtx_21.mtx = share(matrix::Csr<ValueType, IndexType>::create(
            exec, size, std::move(values_tmp), std::move(row_idxs_tmp),
            std::move(col_ptrs_tmp)));
    }

    auto partition_idxs = partitions.get_data();
    const auto values_source = global_mtx->get_const_values();
    const auto row_ptrs_source = global_mtx->get_const_row_ptrs();
    const auto col_idxs_source = global_mtx->get_const_col_idxs();
    auto row_ptrs_current_source = submtx_21.row_ptrs_tmp.get_data();

    auto values_submtx_21 = submtx_21.mtx->get_values();
    auto col_ptrs_submtx_21 = submtx_21.mtx->get_row_ptrs();
    auto row_idxs_submtx_21 = submtx_21.mtx->get_col_idxs();
    auto col_ptrs_current_submtx_21 = submtx_21.col_ptrs_tmp.get_data();
    auto block_col_ptrs_submtx_21 = submtx_21.block_col_ptrs.get_data();

    // Initializes temporary row_pointers to (2, 1) submatrix of global_mtx.
    exec->copy(size[0] + 1, &row_ptrs_source[split_index],
               row_ptrs_current_source);

// Updates col_ptrs[partition_idxs[block]] += block_col_ptrs[block] for each
// 0 <= block < num_blocks.
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto col_idx = partition_idxs[block];
        col_ptrs_submtx_21[col_idx] = block_col_ptrs_submtx_21[block];
    }

    // Performs a reduction on col_ptrs of partition. @Replace with ginkgo
    // partial sum function.
    for (auto block = 0; block < num_blocks; block++) {
        for (auto col = partition_idxs[block] + 1;
             col < partition_idxs[block + 1]; col++) {
            col_ptrs_submtx_21[col] += col_ptrs_submtx_21[col - 1];
        }
    }

// Main loop.
#pragma omp parallel for schedule(dynamic)
    for (auto block_index = 0; block_index < num_blocks; block_index++) {
        const auto col_start = partition_idxs[block_index];
        const auto col_end = partition_idxs[block_index + 1];
        IndexType col_min = partition_idxs[block_index];
        IndexType row_min = split_index;

        // Computes col_ptrs of arrow_submatrix_21.
        // This part calculates the number of nonzeros on each column.
        for (auto row_submtx_21 = row_start; row_submtx_21 < row_end;
             row_submtx_21++) {
            auto row_source = row_submtx_21 - row_start;
            auto row_index_source = row_ptrs_current_source[row_source];
            auto col_source = col_idxs_source[row_index_source];
            if ((col_source >= col_start) && (col_source < col_end)) {
                for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
                     col_submtx_21++) {
                    col_ptrs_current_submtx_21[col_submtx_21 + 1] += 1;
                }
            }
        }

        // Sets row_idxs_submtx_21 for entries in submatrix block.
        auto col_range = col_end - col_start + 1;
        exec->copy(col_range, &col_ptrs_submtx_21[col_start],
                   &col_ptrs_current_submtx_21[col_start]);
        for (auto row = row_start; row < row_end; row++) {
            auto row_index_source = row_ptrs_current_source[row];
            auto col_source = col_idxs_source[row_index_source];
            if ((col_source >= col_start) && (col_source < col_end)) {
                for (auto col_submtx_21 = col_start; col_submtx_21 < col_end;
                     col_submtx_21++) {
                    auto col_index = col_ptrs_current_submtx_21[col_submtx_21];
                    col_ptrs_current_submtx_21[col_submtx_21] += 1;
                    row_idxs_submtx_21[col_index] = row;
                }
            }
        }

        // Copies numerical values from global_mtx to arrow_submatrix_21.mtx.
        // Modifies row_ptrs (of arrow_submatrix_21) while copying.
        exec->copy(col_end - col_start + 1, &col_ptrs_submtx_21[col_start],
                   &col_ptrs_current_submtx_21[col_start]);
        auto remaining_nz = symbolic_count_col_check<IndexType>(
            row_ptrs_current_source, col_idxs_source, 0, row_end - row_start,
            col_end);
        while (remaining_nz > 0) {
            IndexType num_occurences = 0;
            find_min_col(submtx_21, row_ptrs_source, col_idxs_source, row_start,
                         row_end, &col_min, &row_min, &num_occurences);
            auto row_index_source =
                row_ptrs_current_source[row_min - row_start];
            auto col = col_idxs_source[row_index_source];
            auto col_index = col_ptrs_current_submtx_21[col];
            values_submtx_21[col_index] = values_source[row_index_source];
            row_ptrs_current_source[row_min - row_start] += 1;
            col_ptrs_current_submtx_21[col] += 1;
            remaining_nz -= 1;
        }
    }

    // Resets row_ptrs of arrow_submatrix_21 to their original position.
    auto row_ptrs = submtx_21.mtx->get_row_ptrs();
    for (auto i = num_blocks - 1; i >= 0; i--) {
        auto row_end = partition_idxs[i + 1];
        auto row_start = partition_idxs[i];
        for (auto j = row_end; j >= row_start; j--) {
            row_ptrs[j] = (j > 0) ? row_ptrs[j - 1] : 0;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_21_KERNEL);

// Step 3 of computing submatrix_21 of L factor. Sets up the
// nonzeros of submatrix_21 of L factor.
template <typename ValueType, typename IndexType>
void step_3_impl_factorize(
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_partitions<IndexType>& partitions)
{
    using dense = matrix::Dense<ValueType>;
    auto exec = submtx_21.exec;
    auto block_col_ptrs_data = submtx_21.block_col_ptrs.get_data();
    auto partition_idxs = partitions.get_data();
    auto dense_u_factors = submtx_11.dense_u_factors;
    auto nz_per_block = submtx_21.nz_per_block.get_data();
    auto num_blocks = submtx_21.num_blocks;
    array<ValueType> residuals = array<ValueType>(exec, submtx_21.nz);
    exec->copy(submtx_21.nz, submtx_21.mtx->get_values(), residuals.get_data());
#pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        if (nz_per_block[block] > 0) {
            // Sets up dimensions of local data.
            auto stride = 1;
            IndexType row_idx = partition_idxs[block];
            IndexType block_size =
                partition_idxs[block + 1] - partition_idxs[block];
            dim<2> dim_tmp = {static_cast<size_type>(block_size),
                              static_cast<size_type>(block_size)};

            // Computes the left solution to a local linear system.
            dim<2> dim_rhs;
            dim_rhs[0] =
                (block_col_ptrs_data[block + 1] - block_col_ptrs_data[block]) /
                block_size;
            dim_rhs[1] = block_size;
            auto values_u_factor =
                dense_u_factors.get_data()[block].get()->get_values();
            auto values_21 =
                &submtx_21.mtx->get_values()[block_col_ptrs_data[block]];
            // upper_triangular_left_solve_kernel<ValueType, IndexType>(
            //     dim_tmp, values_u_factor, dim_rhs, values_21);

            // Computes residual vectors.
            auto num_elems = dim_rhs[0] * dim_rhs[1];
            auto values_residual =
                &residuals.get_data()[block_col_ptrs_data[block]];
            auto residual_vectors = dense::create(
                exec, dim_rhs,
                array<ValueType>::view(exec, num_elems, values_residual),
                stride);
            dim<2> dim_rnorm = {1, dim_rhs[1]};
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
                    // std::cout << "  abs values: " <<
                    // std::abs(residual_norm->get_values()[i]) << '\n';
                    break;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_3_IMPL_FACTORIZE_21_KERNEL);

// Computes the schur complement of submatrix_22.
template <typename ValueType, typename IndexType>
void step_1_impl_compute_schur_complement(
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_submatrix_22<ValueType, IndexType>& submtx_22,
    factorization::arrow_partitions<IndexType>& partitions)
{
    auto partition_idxs = partitions.get_data();
    dim<2> size = {submtx_21.size[0], submtx_12.size[1]};
    IndexType num_blocks = submtx_11.num_blocks;
    // Reduction over block MM-multiplication.
    // #pragma omp parallel for schedule(dynamic)
    for (auto block = 0; block < num_blocks; block++) {
        auto blk = partition_idxs[block + 1] - partition_idxs[block];
        spdgemm_blocks<ValueType, IndexType>(size, blk, submtx_11, submtx_12,
                                             submtx_21, submtx_22, block, -1);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_1_IMPL_COMPUTE_SCHUR_COMPLEMENT_KERNEL);


template <typename ValueType, typename IndexType>
void spdgemm_blocks(
    dim<2> size, IndexType block_size,
    factorization::arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    factorization::arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    factorization::arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    factorization::arrow_submatrix_22<ValueType, IndexType>& submtx_22,
    IndexType block_index, ValueType alpha)
{
    const auto block_col_ptrs = submtx_21.block_col_ptrs.get_data();
    const auto block_row_ptrs = submtx_12.block_row_ptrs.get_data();
    const auto num_rows_21 =
        (block_col_ptrs[block_index + 1] - block_col_ptrs[block_index]) /
        block_size;
    const auto num_cols_12 =
        (block_row_ptrs[block_index + 1] - block_row_ptrs[block_index]) /
        block_size;
    auto schur_complement_values = submtx_22.schur_complement->get_values();
    const auto values_21 = submtx_21.mtx->get_values();
    auto row_idxs_21 = submtx_21.mtx->get_col_idxs();
    const auto values_12 = submtx_12.mtx->get_values();
    auto col_idxs_12 = submtx_21.mtx->get_col_idxs();
    const auto split_index = submtx_12.split_index;
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

                schur_complement_values[size[1] * row + col] +=
                    (alpha * value_21 * value_12);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL);

// Computes L and U factors of submatrix_22.
template <typename ValueType, typename IndexType>
void step_2_impl_factorize(
    factorization::arrow_submatrix_22<ValueType, IndexType>& submtx_22)
{
    factorize_kernel(submtx_22.schur_complement.get(),
                     submtx_22.dense_l_factor.get(),
                     submtx_22.dense_u_factor.get());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_22_KERNEL);

// Solves triangular system. L-factor is stored in row-major ordering.
template <typename ValueType>
void lower_triangular_solve_kernel(dim<2> dim_l_factor, ValueType* l_factor,
                                   dim<2> dim_rhs, ValueType* rhs_matrix)
{
    // Computes rhs_matri[dim_rhs[1]*row + num_rhs] - dot_product[num_rhs]
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
void upper_triangular_solve_kernel(dim<2> dim_l_factor, ValueType* l_factor,
                                   dim<2> dim_rhs, ValueType* rhs_matrix)
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL_KERNEL);

// Uses column-major ordering
template <typename ValueType>
void upper_triangular_left_solve_kernel(dim<2> dim_l_factor,
                                        ValueType* l_factor, dim<2> dim_lhs,
                                        ValueType* lhs_matrix)
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
        auto col_current = col_idxs[row_index];
        while ((col_current < col_end) && (col_old < col_current)) {
            auto col_local = col_current - col_start;
            values_mtx[num_rows * row_local + col_local] = values[row_index];
            col_old = col_current;
            row_index += 1;
            col_current = col_idxs[row_index];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL);

// Factorize kernels.
template <typename ValueType>
void factorize_kernel(matrix::Dense<ValueType>* mtx,
                      matrix::Dense<ValueType>* l_factor,
                      matrix::Dense<ValueType>* u_factor)
{
    auto mtx_values = mtx->get_values();
    auto l_values = l_factor->get_values();
    auto u_values = u_factor->get_values();

    for (auto r = 0; r < l_factor->get_size()[0]; ++r) {
        for (auto c = 0; c < l_factor->get_size()[1]; ++c) {
            l_values[l_factor->get_size()[0] * r + c] = 0.0;
        }
    }

    for (auto r = 0; r < u_factor->get_size()[0]; ++r) {
        for (auto c = 0; c < u_factor->get_size()[1]; ++c) {
            u_values[u_factor->get_size()[0] * r + c] = 0.0;
        }
    }

    for (auto i = 0; i < mtx->get_size()[0]; ++i) {
        ValueType pivot = mtx_values[mtx->get_size()[0] * i + i];
        if (abs(pivot) < PIVOT_THRESHOLD) {
            pivot += PIVOT_AUGMENTATION;
        }

        // store in row-major format
        l_values[l_factor->get_size()[0] * i + i] = 1.0;
        for (auto j = i + 1; j < mtx->get_size()[0]; ++j) {
            l_values[l_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i] / pivot;
        }

        // store in col-major format
        u_values[u_factor->get_size()[0] * i + i] = pivot;
        for (auto j = i + 1; j < mtx->get_size()[1]; ++j) {
            u_values[u_factor->get_size()[0] * j + i] =
                mtx_values[mtx->get_size()[0] * j + i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FACTORIZE_KERNEL);

// Triangular solve kernels.

template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_1(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    auto partition_idxs = internal_mtx.partitions_.get_data();
    auto dense_l_factors = internal_mtx.submtx_11_.dense_l_factors;
    auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    IndexType num_rhs = 1;  // 1 rhs is used
    for (auto block = 0; block < num_blocks; block++) {
        IndexType block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {static_cast<size_type>(block_size),
                          static_cast<size_type>(block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(block_size),
                          static_cast<size_type>(num_rhs)};
        auto values_l_factor =
            dense_l_factors.get_data()[block].get()->get_values();
        auto values_rhs = &rhs->get_values()[partition_idxs[block] * num_rhs];
        lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                      values_rhs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_1_KERNEL);

template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_2(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    IndexType num_blocks = internal_mtx.submtx_11_.num_blocks;
    auto partition_idxs = internal_mtx.partitions_.get_data();
    auto block_col_ptrs_idxs =
        internal_mtx.submtx_21_.block_col_ptrs.get_data();
    IndexType num_rhs = 1;
    auto values_l_factor = internal_mtx.submtx_21_.mtx.get()->get_values();
    for (auto block = 0; block < num_blocks; block++) {
        auto col_idx = block_col_ptrs_idxs[block];
        IndexType block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {
            static_cast<size_type>(
                (block_col_ptrs_idxs[block + 1] - block_col_ptrs_idxs[block]) /
                block_size),
            static_cast<size_type>(block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(num_rhs),
                          static_cast<size_type>(block_size)};
        auto values_21 = &values_l_factor[col_idx];
        auto values_rhs = &rhs->get_values()[partition_idxs[block]];
        // csc_spdgemm<ValueType, IndexType>(dim_tmp, values_l_factor, dim_rhs,
        //                                     values_rhs, -1.0);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_2_KERNEL);

template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_3(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    // auto mtx = internal_mtx.submtx_22;
    auto split_index = internal_mtx.submtx_22_.split_index;
    IndexType num_rhs = 1;
    dim<2> dim_tmp = internal_mtx.submtx_22_.size;
    dim<2> dim_rhs = {dim_tmp[0], static_cast<size_type>(num_rhs)};
    auto values_l_factor =
        internal_mtx.submtx_22_.dense_l_factor.get()->get_values();
    auto values_rhs = &rhs->get_values()[split_index];
    lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                  values_rhs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_3_KERNEL);

template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_1(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    // auto mtx = internal_mtx.submtx_22;
    auto split_index = internal_mtx.submtx_22_.split_index;
    IndexType num_rhs = 1;
    dim<2> dim_tmp = internal_mtx.submtx_22_.size;
    dim<2> dim_rhs = {dim_tmp[0], static_cast<size_type>(num_rhs)};
    auto values_u_factor =
        internal_mtx.submtx_22_.dense_u_factor.get()->get_values();
    auto values_rhs = &rhs->get_values()[split_index];
    lower_triangular_solve_kernel(dim_tmp, values_u_factor, dim_rhs,
                                  values_rhs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_1_KERNEL);

template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_2(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    IndexType num_blocks = internal_mtx.submtx_11_.num_blocks;
    auto partition_idxs = internal_mtx.partitions_.get_data();
    auto block_row_ptrs_idxs =
        internal_mtx.submtx_12_.block_row_ptrs.get_data();
    auto split_index = internal_mtx.submtx_12_.split_index;
    IndexType num_rhs = 1;
    auto values_u_factor = internal_mtx.submtx_12_.mtx.get()->get_values();
    for (auto block = 0; block < num_blocks; block++) {
        auto row_idx = block_row_ptrs_idxs[block];
        IndexType block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {
            static_cast<size_type>(block_size),
            static_cast<size_type>(
                (block_row_ptrs_idxs[block + 1] - block_row_ptrs_idxs[block]) /
                block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(block_size),
                          static_cast<size_type>(num_rhs)};
        auto values_12 = &values_u_factor[row_idx];
        auto values_rhs = &rhs->get_values()[split_index];
        // csr_spdgemm<ValueType, IndexType>(dim_tmp, values_12, dim_rhs,
        //                                     values_rhs, -1.0);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_2_KERNEL);

template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_3(
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    auto partition_idxs = internal_mtx.partitions_.get_data();
    auto dense_u_factors = internal_mtx.submtx_11_.dense_u_factors;
    auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    IndexType num_rhs = 1;  // 1 rhs is used
    for (auto block = 0; block < num_blocks; block++) {
        IndexType block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {static_cast<size_type>(block_size),
                          static_cast<size_type>(block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(block_size),
                          static_cast<size_type>(num_rhs)};
        auto values_u_factor =
            dense_u_factors.get_data()[block].get()->get_values();
        auto values_rhs = &rhs->get_values()[partition_idxs[block] * num_rhs];
        lower_triangular_solve_kernel(dim_tmp, values_u_factor, dim_rhs,
                                      values_rhs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_3_KERNEL);

}  // namespace arrow_lu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
