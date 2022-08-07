#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>

#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"

namespace gko {
namespace kernels {
namespace reference {
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
    preprocess_submatrix_12(exec, partitions, mtx, submtx_12,
                            row_ptrs_src_cur_array, row_ptrs_dst_cur_array);
    initialize_submatrix_12(exec, partitions, mtx, submtx_12,
                            row_ptrs_src_cur_array);
    factorize_submatrix_12(exec, partitions, submtx_11, submtx_12);
    preprocess_submatrix_21(exec, partitions, mtx, submtx_21,
                            col_ptrs_dst_cur_array, row_ptrs_src_cur_array);
    initialize_submatrix_21(exec, workspace->get_partitions(), mtx,
                            workspace->get_submatrix_21());
    factorize_submatrix_21(exec, workspace->get_partitions(),
                           workspace->get_submatrix_11(),
                           workspace->get_submatrix_21());
    initialize_submatrix_22(exec, partitions, submtx_11, submtx_12, submtx_21,
                            submtx_22);
    factorize_submatrix_22(exec, workspace->get_submatrix_22());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);


// Checks rows [row_start, ..., row_end] of submatrix_12 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <typename IndexType>
void find_min_col(const IndexType* row_ptrs_src, const IndexType* col_idxs_src,
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
        auto col = col_idxs_src[row_index];
        num_occurences =
            ((col == col_min) && (row_index < row_ptrs_src[row + 1]))
                ? (num_occurences + 1)
                : num_occurences;
        row_min = ((col < col_min) && (row_index < row_ptrs_src[row + 1]))
                      ? row
                      : row_min;
        col_min = ((col < col_min) && (row_index < row_ptrs_src[row + 1]))
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
IndexType compute_remaining_nnz_row_check(const IndexType* row_ptrs_src,
                                          IndexType* row_ptrs_cur,
                                          IndexType row_start,
                                          IndexType row_end)
{
    auto remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        remaining_nnz += ((row_ptrs_src[row + 1] > row_ptrs_cur[row])
                              ? row_ptrs_src[row + 1] - row_ptrs_cur[row]
                              : 0);
        // std::cout << "row_ptrs_src[row + 1]: " << row_ptrs_src[row + 1] << ",
        // row_ptrs_cur[row]: " << row_ptrs_cur[row] << '\n';
    }
    return remaining_nnz;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_REMAINING_NNZ_ROW_CHECK_KERNEL);


// Checks if there is a nonzero entry in rows row_start, ..., row_end of input
// matrix. Performs the checks by comparing column indices. Returns the number
// of remaining nzs.
template <typename IndexType>
IndexType compute_remaining_nnz_col_check(const IndexType* col_idxs_src,
                                          IndexType* row_ptrs_cur,
                                          IndexType row_start,
                                          IndexType row_end, IndexType col_end)
{
    IndexType remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        auto row_index = row_ptrs_cur[row];
        while ((col_idxs_src[row_index] < col_end) &&
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


template <typename ValueType, typename IndexType>
void allocate_l_factors(std::shared_ptr<const DefaultExecutor> exec,
                        size_type num_blocks, const IndexType* partition_idxs,
                        std::vector<std::unique_ptr<gko::LinOp>>& l_factors_in)
{
    using dense = matrix::Dense<ValueType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    size_type stride = 1;
    auto l_factors = l_factors_in.begin();
    for (auto block = 0; block < num_blocks; block++) {
        const dim<2> block_size = {
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block]),
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block])};
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        l_factors_in.push_back(std::move(
            dense::create(exec, block_size, std::move(tmp_array), stride)));
    }
}


template <typename ValueType, typename IndexType>
void allocate_diag_blocks(std::shared_ptr<const DefaultExecutor> exec,
                          size_type num_blocks, const IndexType* partition_idxs,
                          std::vector<std::unique_ptr<gko::LinOp>>& diag_blocks)
{
    using dense = matrix::Dense<ValueType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    size_type stride = 1;
    for (auto block = 0; block < num_blocks; block++) {
        const dim<2> block_size = {
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block]),
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block])};
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        diag_blocks.push_back(std::move(
            dense::create(exec, block_size, std::move(tmp_array), stride)));
    }
}


template <typename ValueType, typename IndexType>
void allocate_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                        size_type num_blocks, const IndexType* partition_idxs,
                        std::vector<std::unique_ptr<gko::LinOp>>& u_factors_in)
{
    using dense = matrix::Dense<ValueType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    size_type stride = 1;
    auto u_factors = u_factors_in.begin();
    for (auto block = 0; block < num_blocks; block++) {
        const dim<2> block_size = {
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block]),
            static_cast<size_type>(partition_idxs[block + 1] -
                                   partition_idxs[block])};
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);
        u_factors_in.push_back(std::move(
            dense::create(exec, block_size, std::move(tmp_array), stride)));
    }
}


template <typename ValueType, typename IndexType>
void initialize_submatrix_11(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11)
{
    const auto partition_idxs = partitions->get_const_data();
    const auto num_blocks = submtx_11->num_blocks;
    allocate_diag_blocks<ValueType, IndexType>(exec, num_blocks, partition_idxs,
                                               submtx_11->diag_blocks);
    allocate_l_factors<ValueType, IndexType>(exec, num_blocks, partition_idxs,
                                             submtx_11->l_factors);
    allocate_u_factors<ValueType, IndexType>(exec, num_blocks, partition_idxs,
                                             submtx_11->u_factors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INITIALIZE_SUBMATRIX_11_KERNEL);

template <typename ValueType, typename IndexType>
void factorize_diag_blocks(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const IndexType* partition_idxs, const IndexType* row_ptrs,
    const IndexType* col_idxs, const ValueType* values, size_type* nnz_l_out,
    size_type* nnz_u_out, std::vector<std::unique_ptr<gko::LinOp>>& diag_blocks,
    std::vector<std::unique_ptr<gko::LinOp>>& l_factors,
    std::vector<std::unique_ptr<gko::LinOp>>& u_factors)
{
    using dense = matrix::Dense<ValueType>;
    IndexType nnz_l = 0;
    IndexType nnz_u = 0;
    for (auto block = 0; block < num_blocks; block++) {
        const auto len = static_cast<size_type>(partition_idxs[block + 1] -
                                                partition_idxs[block]);
        const dim<2> block_size = {len, len};
        const auto num_elems_dense = static_cast<size_type>(len * len);
        nnz_l += (len * len + len) / 2;
        nnz_u += (len * len + len) / 2;

        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        auto tmp_array = array<ValueType>(exec, block_size[0] * block_size[1]);
        tmp_array.fill(0.0);

        auto system_mtx = share(dense::create(exec));
        as<ConvertibleTo<dense>>(diag_blocks[block].get())
            ->convert_to(system_mtx.get());

        convert_csr_2_dense<ValueType, IndexType>(
            block_size, row_ptrs, col_idxs, values, system_mtx.get(), row_start,
            row_end);

        auto l_factor = share(dense::create(exec));
        as<ConvertibleTo<dense>>(l_factors[block].get())
            ->convert_to(l_factor.get());

        auto u_factor = share(dense::create(exec));
        as<ConvertibleTo<dense>>(u_factors[block].get())
            ->convert_to(u_factor.get());

        factorize_kernel(system_mtx.get(), l_factor.get(), u_factor.get());
    }
    *nnz_l_out = nnz_l;
    *nnz_u_out = nnz_u;
}


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
    const auto values_src = mtx->get_const_values();
    const auto col_idxs_src = mtx->get_const_col_idxs();
    const auto row_ptrs_src = mtx->get_const_row_ptrs();
    auto row_ptrs = submtx_11->row_ptrs_cur.get_data();
    exec->copy(split_index + 1, row_ptrs_src, row_ptrs);
    factorize_diag_blocks(exec, num_blocks, partition_idxs, row_ptrs,
                          col_idxs_src, values_src, &submtx_11->nnz_l,
                          &submtx_11->nnz_u, submtx_11->diag_blocks,
                          submtx_11->l_factors, submtx_11->u_factors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL);


template <typename IndexType>
void move_ptrs_to_submatrix_12(const IndexType* col_idxs_src,
                               IndexType* row_ptrs_cur, IndexType row_start,
                               IndexType row_end, size_type split_index)
{
    for (auto row = row_start; row < row_end; row++) {
        while ((col_idxs_src[row_ptrs_cur[row]] < split_index) &&
               (row_ptrs_cur[row] < row_ptrs_cur[row + 1])) {
            row_ptrs_cur[row] += 1;
        }
    }
}

template <typename IndexType>
void move_min_wavefront(const IndexType* row_ptrs_src,
                        const IndexType* col_idxs_src,
                        IndexType* row_ptrs_cur_src,
                        IndexType* row_ptrs_cur_submtx, IndexType row_start,
                        IndexType row_end, IndexType row_min, IndexType col_min,
                        size_type split_index, size_type len,
                        IndexType* nnz_out, IndexType* remaining_nnz_out)
{
    bool found_nonzero_column = false;
    auto nnz = *nnz_out;
    auto remaining_nnz = *remaining_nnz_out;
    for (auto row = row_start; row < row_end; row++) {
        const auto row_index = row_ptrs_cur_src[row];
        const auto col = col_idxs_src[row_index];
        if ((col == col_min) &&
            (row_ptrs_cur_src[row] < row_ptrs_src[row + 1])) {
            row_ptrs_cur_src[row] += 1;
            remaining_nnz -= 1;
            found_nonzero_column = true;
        }

        if ((row == row_min) && (row_ptrs_cur_submtx[row + 1] == 0)) {
            row_ptrs_cur_submtx[row + 1] += len;
        }
    }

    nnz = (found_nonzero_column && (col_min >= split_index)) ? nnz + len : nnz;

    *remaining_nnz_out = remaining_nnz;
    *nnz_out = nnz;
}

template <typename IndexType>
void convert_row_ptrs_to_global(size_type num_blocks,
                                const IndexType* partition_idxs,
                                const IndexType* block_row_ptrs,
                                IndexType* row_ptrs_tmp)
{
    for (auto block = 0; block < num_blocks; block++) {
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        for (auto row = row_start; row < row_end; row++) {
            row_ptrs_tmp[row + 1] += row_ptrs_tmp[row];
        }
    }
}


// Step 1 of computing LU factors of submatrix_12. Computes the number of
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
    const auto max_col = submtx_12->size[0] + submtx_12->size[1] + 1;
    const auto num_blocks = submtx_12->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto row_ptrs_src = mtx->get_const_row_ptrs();
    const auto col_idxs_src = mtx->get_const_col_idxs();
    auto row_ptrs_cur_src = row_ptrs_cur_src_array.get_data();
    auto block_row_ptrs = submtx_12->block_ptrs.get_data();
    auto nnz_per_block = submtx_12->nnz_per_block.get_data();
    auto row_ptrs_cur_submtx = row_ptrs_cur_dst_array.get_data();
    IndexType nnz_tmp = 0;
    IndexType col_min = 0;
    IndexType row_min = 0;
    IndexType num_occurences = 0;
    row_ptrs_cur_dst_array.fill(0);
    exec->copy(submtx_12->size[0], row_ptrs_src, row_ptrs_cur_src);

    for (auto block = 0; block < num_blocks; block++) {
        const auto len = partition_idxs[block + 1] - partition_idxs[block];
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];

        move_ptrs_to_submatrix_12(col_idxs_src, row_ptrs_cur_src, row_start,
                                  row_end, submtx_12->split_index);

        auto remaining_nnz = compute_remaining_nnz_row_check(
            row_ptrs_src, row_ptrs_cur_src, row_start, row_end);
        nnz_per_block[block] = remaining_nnz;

        while (remaining_nnz > 0) {
            find_min_col(row_ptrs_src, col_idxs_src, row_ptrs_cur_src,
                         submtx_12->size, row_start, row_end, &col_min,
                         &row_min, &num_occurences);

            move_min_wavefront(row_ptrs_src, col_idxs_src, row_ptrs_cur_src,
                               row_ptrs_cur_submtx, row_start, row_end, row_min,
                               col_min, submtx_12->split_index, len, &nnz_tmp,
                               &remaining_nnz);
            col_min = max_col;
        }

        block_row_ptrs[block + 1] = nnz_tmp;
    }

    submtx_12->nnz = nnz_tmp;

    convert_row_ptrs_to_global(num_blocks, partition_idxs, block_row_ptrs,
                               row_ptrs_cur_submtx);

    {
        array<IndexType> col_idxs_tmp = {
            exec, static_cast<size_type>(submtx_12->nnz)};
        array<ValueType> values_tmp = {exec,
                                       static_cast<size_type>(submtx_12->nnz)};
        submtx_12->mtx = share(matrix::Csr<ValueType, IndexType>::create(
            exec, submtx_12->size, std::move(values_tmp),
            std::move(col_idxs_tmp), std::move(row_ptrs_cur_dst_array)));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PREPROCESS_SUBMATRIX_12_KERNEL);


template <typename ValueType, typename IndexType>
IndexType copy_submatrix_12_block(IndexType* row_ptrs_cur_src,
                                  IndexType* row_ptrs_cur_submtx,
                                  const IndexType* col_idxs_src,
                                  const ValueType* values_src,
                                  IndexType* col_idxs_submtx,
                                  ValueType* values_submtx, IndexType row_start,
                                  IndexType row_end, size_type split_index,
                                  IndexType col_min, IndexType remaining_nnz)
{
    for (auto row = row_start; row < row_end; row++) {
        const auto row_index_src = row_ptrs_cur_src[row];
        const auto col = col_idxs_src[row_index_src];
        const auto row_index_submtx = row_ptrs_cur_submtx[row];

        values_submtx[row_index_submtx] =
            (col == col_min) ? values_src[row_index_src] : 0.0;
        col_idxs_submtx[row_index_submtx] =
            col_min - static_cast<IndexType>(split_index);
        row_ptrs_cur_submtx[row] += 1;
        if (col == col_min) {
            remaining_nnz -= 1;
            row_ptrs_cur_src[row] += 1;
        }
    }
    return remaining_nnz;
}

// Resets row_ptrs to original position.
template <typename IndexType>
void reset_row_ptrs(size_type num_blocks, const IndexType* partition_idxs,
                    IndexType* row_ptrs)
{
    for (auto block = static_cast<IndexType>(num_blocks) - 1; block >= 0;
         block--) {
        const auto row_end = partition_idxs[block + 1];
        const auto row_start = partition_idxs[block];
        for (auto row = row_end; row >= row_start; row--) {
            row_ptrs[row] = (row > 0) ? row_ptrs[row - 1] : 0;
        }
    }
}

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
    const auto row_ptrs_src = mtx->get_const_row_ptrs();
    const auto partition_idxs = partitions->get_const_data();
    const auto values_src = mtx->get_const_values();
    const auto col_idxs_src = mtx->get_const_col_idxs();
    auto row_ptrs_cur_src = row_ptrs_cur_src_array.get_data();
    auto values = submtx_12->mtx->get_values();
    auto col_idxs = submtx_12->mtx->get_col_idxs();
    auto row_ptrs = submtx_12->mtx->get_row_ptrs();
    exec->copy(submtx_12->size[0] + 1, row_ptrs_src, row_ptrs_cur_src);

    for (auto block = 0; block < 19; block++) {
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        const auto row_start = partition_idxs[block];
        const auto row_end = partition_idxs[block + 1];
        move_ptrs_to_submatrix_12(col_idxs_src, row_ptrs_cur_src, row_start,
                                  row_end, submtx_12->split_index);
        IndexType num_occurences = 0;
        IndexType col_min = 0;
        IndexType row_min = 0;
        IndexType remaining_nnz = 0;

        remaining_nnz = compute_remaining_nnz_row_check(
            row_ptrs_src, row_ptrs_cur_src, row_start, row_end);

        while (remaining_nnz > 0) {
            find_min_col(row_ptrs_src, col_idxs_src, row_ptrs_cur_src,
                         submtx_12->size, row_start, row_end, &col_min,
                         &row_min, &num_occurences);

            remaining_nnz = copy_submatrix_12_block(
                row_ptrs_cur_src, row_ptrs, col_idxs_src, values_src, col_idxs,
                values, row_start, row_end, split_index, col_min,
                remaining_nnz);
        }
    }
    reset_row_ptrs(num_blocks, partition_idxs, row_ptrs);
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
    for (auto block = 0; block < num_blocks; block++) {
        if (submtx_12->nnz_per_block.get_data()[block] > 0) {
            const auto block_size = static_cast<size_type>(
                partition_idxs[block + 1] - partition_idxs[block]);
            const auto num_cols =
                static_cast<size_type>(block_row_ptrs[block + 1] -
                                       block_row_ptrs[block]) /
                block_size;
            const dim<2> dim_tmp = {static_cast<size_type>(block_size),
                                    static_cast<size_type>(block_size)};
            const dim<2> dim_rhs = {block_size, num_cols};

            auto l_factor = share(dense::create(exec));
            as<ConvertibleTo<dense>>(submtx_11->l_factors[block].get())
                ->convert_to(l_factor.get());
            auto values_l_factor = l_factor->get_values();
            auto values_12 =
                &submtx_12->mtx->get_values()[block_row_ptrs[block]];
            lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                          values_12);

            auto num_elems = dim_rhs[0] * dim_rhs[1];
            auto values_residual = &residuals.get_data()[block_row_ptrs[block]];
            auto residual_vectors = dense::create(
                exec, dim_rhs,
                array<ValueType>::view(exec, num_elems, values_residual),
                stride);

            dim<2> dim_rnorm = {1, dim_rhs[1]};
            array<ValueType> values_rnorm = {exec, dim_rnorm[1]};
            values_rnorm.fill(0.0);
            auto residual_norm =
                dense::create(exec, dim_rnorm, values_rnorm, stride);

            auto solution =
                dense::create(exec, dim_rhs,
                              array<ValueType>::view(
                                  exec, dim_rhs[0] * dim_rhs[1], values_12),
                              stride);

            // Performs MM multiplication.
            auto x = solution->get_values();
            auto b = residual_vectors->get_values();
            auto l_vals = l_factor->get_values();
            for (auto row_l = 0; row_l < dim_tmp[0]; row_l++) {
                for (auto col_b = 0; col_b < dim_rhs[1]; col_b++) {
                    for (auto row_b = 0; row_b < dim_rhs[0]; row_b++) {
                        b[dim_rhs[1] * row_l + col_b] -=
                            l_vals[dim_tmp[1] * row_l + row_b] *
                            x[dim_rhs[1] * row_b + col_b];
                    }
                }
            }

            // Computes residual norms.
            auto r = residual_vectors.get();
            r->compute_norm2(residual_norm.get());
            for (auto i = 0; i < residual_norm->get_size()[1]; ++i) {
                if (std::abs(residual_norm->get_values()[i]) > 1e-8) {
                    std::cout << "i: " << i << "abs values: "
                              << std::abs(residual_norm->get_values()[i])
                              << ", block_index: " << block << '\n';
                    break;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL);


template <typename IndexType>
void preprocess_submatrix_21_block(dim<2> size, const IndexType* partition_idxs,
                                   const IndexType* col_idxs_src,
                                   IndexType* row_ptrs_cur_submtx,
                                   IndexType* block_col_ptrs,
                                   IndexType* col_ptrs,
                                   IndexType* nnz_per_block, IndexType block,
                                   IndexType* nnz_out)
{
    IndexType max = 0;
    IndexType nnz = 0;
    for (auto row = 0; row < size[0]; row++) {
        auto col_index = row_ptrs_cur_submtx[row];
        auto col = col_idxs_src[col_index];
        const auto blk = partition_idxs[block + 1] - partition_idxs[block];
        const auto col_start = partition_idxs[block];
        const auto col_end = partition_idxs[block + 1];
        max = (col > max) ? col : max;
        if ((col >= partition_idxs[block]) &&
            (col < partition_idxs[block + 1])) {
            nnz += blk;
            if (block == 6) {
                std::cout << "row: " << row << ", col: " << col
                          << ", col_start: " << col_start
                          << ", col_end: " << col_end << '\n';
                std::cout << "col_ptrs[21]: " << col_ptrs[21] << '\n';
                std::cout << "col_ptrs[22]: " << col_ptrs[22] << '\n';
                std::cout << "col_ptrs[23]: " << col_ptrs[23] << '\n';
                std::cout << "col_ptrs[24]: " << col_ptrs[24] << '\n';
            }
            for (auto c = col_start; c < col_end; c++) {
                col_ptrs[c + 1] += 1;
                if (block == 6)
                    std::cout << "c+1: " << c + 1
                              << ", col_ptrs[c + 1]: " << col_ptrs[c + 1]
                              << '\n';
            }
            if (block == 6) std::cout << "\n";

            row_ptrs_cur_submtx[row] += 1;
        }
    }
    *nnz_out += nnz;
    block_col_ptrs[block + 1] = *nnz_out;
    nnz_per_block[block] = *nnz_out;
}


// This part computes a partial sum of the nonzero/column to form
// the col_ptrs array of arrow_submatrix_21.
template <typename IndexType>
void compute_global_col_ptrs(IndexType* col_ptrs, IndexType* block_col_ptrs,
                             IndexType block_index, IndexType col_start,
                             IndexType col_end)
{
    col_ptrs[col_start] = block_col_ptrs[block_index];
    for (auto col = col_start + 1; col <= col_end; col++) {
        col_ptrs[col] += col_ptrs[col - 1];
    }
}


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
    const auto size = submtx_21->size;
    const auto split_index = submtx_21->split_index;
    const auto num_blocks = submtx_21->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto col_idxs_src = mtx->get_const_col_idxs();
    const auto row_ptrs_src = mtx->get_const_row_ptrs();
    const auto values_src = mtx->get_const_values();
    auto col_ptrs = col_ptrs_dst_array.get_data();
    auto nnz_per_block = submtx_21->nnz_per_block.get_data();
    auto row_ptrs_cur_submtx = row_ptrs_dst_array.get_data();
    auto block_col_ptrs = submtx_21->block_ptrs.get_data();
    IndexType nnz = 0;
    exec->copy(size[0] + 1, &row_ptrs_src[split_index], row_ptrs_cur_submtx);
    for (auto block = 0; block < num_blocks; block++) {
        const auto col_start = partition_idxs[block];
        const auto col_end = partition_idxs[block + 1];
        preprocess_submatrix_21_block(submtx_21->size, partition_idxs,
                                      col_idxs_src, row_ptrs_cur_submtx,
                                      block_col_ptrs, col_ptrs, nnz_per_block,
                                      static_cast<IndexType>(block), &nnz);
    }
    submtx_21->nnz = nnz;

    for (auto block = 0; block < num_blocks; block++) {
        const auto col_start = partition_idxs[block];
        const auto col_end = partition_idxs[block + 1];
        compute_global_col_ptrs(col_ptrs, block_col_ptrs,
                                static_cast<IndexType>(block), col_start,
                                col_end);
    }
    std::cout << "col_ptrs[0]: " << col_ptrs[0] << '\n';
    std::cout << "col_ptrs[1]: " << col_ptrs[1] << '\n';
    std::cout << "col_ptrs[2]: " << col_ptrs[2] << '\n';
    std::cout << "col_ptrs[3]: " << col_ptrs[3] << '\n';
    std::cout << '\n';
    std::cout << "col_ptrs[21]: " << col_ptrs[21] << '\n';
    std::cout << "col_ptrs[22]: " << col_ptrs[22] << '\n';
    std::cout << "col_ptrs[23]: " << col_ptrs[23] << '\n';
    std::cout << "col_ptrs[24]: " << col_ptrs[24] << '\n';
    {
        const dim<2> size_tmp = {submtx_21->size[1], submtx_21->size[0]};
        auto row_idxs_tmp =
            array<IndexType>(exec, static_cast<size_type>(submtx_21->nnz));
        auto values_tmp =
            array<ValueType>(exec, static_cast<size_type>(submtx_21->nnz));
        row_idxs_tmp.fill(0);
        values_tmp.fill(0.0);
        submtx_21->mtx = share(matrix::Csr<ValueType, IndexType>::create(
            exec, size_tmp, std::move(values_tmp), std::move(row_idxs_tmp),
            std::move(col_ptrs_dst_array)));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PREPROCESS_SUBMATRIX_21_KERNEL);


// Computes col_ptrs of arrow_submatrix_21.
// This part calculates the number of nonzeros on each column.
template <typename IndexType>
void compute_col_ptrs_submatrix_21(IndexType* row_ptrs_cur,
                                   const IndexType* col_idxs_src,
                                   IndexType* col_ptrs, IndexType row_start,
                                   IndexType row_end, IndexType col_start,
                                   IndexType col_end)
{
    for (auto row = row_start; row < row_end; row++) {
        const auto row_index_src = row_ptrs_cur[row - row_start];
        const auto col_src = col_idxs_src[row_index_src];
        if ((col_src >= col_start) && (col_src < col_end)) {
            for (auto col = col_start; col < col_end; col++) {
                col_ptrs[col + 1] += 1;
            }
        }
    }
}


// Sets row_idxs of arrow_submatrix_21.
template <typename IndexType>
void set_row_ptrs_submatrix_21(std::shared_ptr<const DefaultExecutor> exec,
                               IndexType* col_ptrs, IndexType* col_ptrs_cur,
                               IndexType* row_ptrs_cur,
                               const IndexType* col_idxs_src,
                               IndexType* row_idxs, IndexType col_start,
                               IndexType col_end, IndexType row_start,
                               IndexType row_end)
{
    exec->copy(col_end - col_start + 1, &col_ptrs[col_start],
               &col_ptrs_cur[col_start]);
    for (auto row_src = row_start; row_src < row_end; row_src++) {
        const auto row_index_src = row_ptrs_cur[row_src - row_start];
        const auto col_src = col_idxs_src[row_index_src];
        if ((col_src >= col_start) && (col_src < col_end)) {
            for (auto col = col_start; col < col_end; col++) {
                const auto col_index_src = col_ptrs_cur[col];
                row_idxs[col_index_src] = row_src;
                col_ptrs_cur[col] += 1;
                // std::cout << "col_index_src: " << col_index_src << ", col: "
                // << col << '\n';
            }
            row_ptrs_cur[row_src - row_start] += 1;
            // std::cout << "\n";
        }
    }
    // std::cout << "\n";
}

// Copies numerical values from mtx to arrow_submatrix_21.mtx.
// Modifies row_ptrs (of arrow_submatrix_21) while copying.
template <typename ValueType, typename IndexType>
void copy_values_submatrix_21(dim<2> size, size_type split_index,
                              std::shared_ptr<const DefaultExecutor> exec,
                              IndexType* col_ptrs, IndexType* col_ptrs_cur,
                              IndexType* row_ptrs_cur, IndexType* row_ptrs_cur2,
                              const IndexType* col_idxs_src,
                              IndexType* row_idxs,
                              const IndexType* row_ptrs_src, ValueType* values,
                              const ValueType* values_src, IndexType col_start,
                              IndexType col_end, IndexType row_start,
                              IndexType row_end)
{
    IndexType col_min = col_start;
    IndexType row_min = static_cast<IndexType>(split_index);
    IndexType num_occurences = 0;
    exec->copy(col_end - col_start + 1, &col_ptrs[col_start],
               &col_ptrs_cur[col_start]);
    auto remaining_nz = compute_remaining_nnz_col_check(
        col_idxs_src, row_ptrs_cur2, static_cast<IndexType>(0),
        row_end - row_start, col_end);
    while (remaining_nz > 0) {
        find_min_col(row_ptrs_src, col_idxs_src, row_ptrs_cur, size, row_start,
                     row_end, &col_min, &row_min, &num_occurences);
        const auto row_index_src = row_ptrs_cur[row_min - row_start];
        const auto col_src = col_idxs_src[row_index_src];
        const auto col_index = col_ptrs_cur[col_src];
        values[col_index] = values_src[row_index_src];
        row_ptrs_cur[row_min - row_start] += 1;
        col_ptrs_cur[col_min] += 1;
        remaining_nz -= 1;
    }
}

// Resets row_ptrs of arrow_submatrix_21 to their original position.
template <typename IndexType>
void reset_col_ptrs_submatrix_21(size_type num_blocks,
                                 const IndexType* partition_idxs,
                                 IndexType* col_ptrs)
{
    for (auto block = static_cast<IndexType>(num_blocks) - 1; block >= 0;
         block--) {
        const auto row_end = partition_idxs[block + 1];
        const auto row_start = partition_idxs[block];
        for (auto row = row_end; row >= row_start; row--) {
            col_ptrs[row] = (row > 0) ? col_ptrs[row - 1] : 0;
        }
    }
}

// Step 2 of computing LU factors of submatrix_21. Sets up the
// nonzeros of submatrix_21 of U factor.
template <typename ValueType, typename IndexType>
void initialize_submatrix_21(
    std::shared_ptr<const DefaultExecutor> exec,
    const factorization::arrow_partitions<IndexType>* partitions,
    const matrix::Csr<ValueType, IndexType>* mtx,
    factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21)
{
    const auto split_index = submtx_21->split_index;
    const auto num_blocks = submtx_21->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    const auto row_ptrs_src = mtx->get_const_row_ptrs();
    const auto col_idxs_src = mtx->get_const_col_idxs();
    const auto values_src = mtx->get_const_values();
    const IndexType row_start = static_cast<IndexType>(submtx_21->split_index);
    const IndexType row_end = static_cast<IndexType>(mtx->get_size()[0]);
    auto values = submtx_21->mtx->get_values();
    auto col_ptrs = submtx_21->mtx->get_row_ptrs();
    auto row_idxs = submtx_21->mtx->get_col_idxs();
    auto row_ptrs_cur = submtx_21->row_ptrs_cur.get_data();
    auto row_ptrs_cur2 = submtx_21->row_ptrs_cur2.get_data();
    auto col_ptrs_cur = submtx_21->col_ptrs_cur.get_data();
    auto block_col_ptrs = submtx_21->block_ptrs.get_data();
    IndexType num_rhs = 0;
    exec->copy(submtx_21->size[0] + 1, &row_ptrs_src[split_index],
               row_ptrs_cur);
    exec->copy(submtx_21->size[0] + 1, &row_ptrs_src[split_index],
               row_ptrs_cur2);
    exec->copy(submtx_21->size[0] + 1, submtx_21->mtx->get_row_ptrs(),
               col_ptrs_cur);

    for (IndexType block_index = 0; block_index < num_blocks; block_index++) {
        const auto col_start = partition_idxs[block_index];
        const auto col_end = partition_idxs[block_index + 1];
        std::cout << "block_index: " << block_index << '\n';
        set_row_ptrs_submatrix_21(exec, col_ptrs, col_ptrs_cur, row_ptrs_cur,
                                  col_idxs_src, row_idxs, col_start, col_end,
                                  row_start, row_end);

        copy_values_submatrix_21(submtx_21->size, submtx_21->split_index, exec,
                                 col_ptrs, col_ptrs_cur, row_ptrs_cur,
                                 row_ptrs_cur2, col_idxs_src, row_idxs,
                                 row_ptrs_src, values, values_src, col_start,
                                 col_end, row_start, row_end);
    }
    reset_col_ptrs_submatrix_21(num_blocks, partition_idxs, col_ptrs);

    std::cout << "row_idxs[9]: " << row_idxs[9] << '\n';
    std::cout << "row_idxs[10]: " << row_idxs[10] << '\n';
    std::cout << "row_idxs[11]: " << row_idxs[11] << '\n';
    std::cout << "row_idxs[53]: " << row_idxs[53] << '\n';
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
    const auto partition_idxs = partitions->get_const_data();
    const auto block_col_ptrs = submtx_21->block_ptrs.get_const_data();
    const auto nnz_per_block = submtx_21->nnz_per_block.get_const_data();
    array<ValueType> residuals = array<ValueType>(exec, submtx_21->nnz);
    exec->copy(submtx_21->nnz, submtx_21->mtx->get_values(),
               residuals.get_data());
    for (auto block = 0; block < num_blocks; block++) {
        if (nnz_per_block[block] > 0) {
            // Sets up dimensions of local data.
            const auto stride = 1;
            const auto row_idx = partition_idxs[block];
            const auto block_size = static_cast<size_type>(
                partition_idxs[block + 1] - partition_idxs[block]);
            const dim<2> dim_tmp = {static_cast<size_type>(block_size),
                                    static_cast<size_type>(block_size)};

            // Computes the left solution to a local linear system.
            const dim<2> dim_rhs = {
                (static_cast<size_type>(block_col_ptrs[block + 1] -
                                        block_col_ptrs[block]) /
                 block_size),
                static_cast<size_type>(block_size)};

            auto system_mtx = share(dense::create(exec));
            as<ConvertibleTo<dense>>(submtx_11->u_factors[block].get())
                ->convert_to(system_mtx.get());
            auto values_u_factor = system_mtx->get_values();
            auto values_submtx_21 =
                &submtx_21->mtx->get_values()[block_col_ptrs[block]];
            upper_triangular_left_solve_kernel(dim_tmp, values_u_factor,
                                               dim_rhs, values_submtx_21);

            // Computes residual vectors.
            const auto num_elems = dim_rhs[0] * dim_rhs[1];
            auto values_residual = &residuals.get_data()[block_col_ptrs[block]];
            auto residual_vectors = dense::create(
                exec, dim_rhs,
                array<ValueType>::view(exec, num_elems, values_residual),
                stride);
            const dim<2> dim_rnorm = {1, dim_rhs[1]};
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
            auto solution = dense::create(
                exec, dim_rhs,
                array<ValueType>::view(exec, dim_rhs[0] * dim_rhs[1],
                                       values_submtx_21),
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
            for (auto i = 0; i < residual_norm->get_size()[1]; i++) {
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
    const dim<2> size = {submtx_21->size[0], submtx_12->size[1]};
    const auto num_blocks = submtx_11->num_blocks;
    const auto partition_idxs = partitions->get_const_data();
    auto schur_complement = share(dense::create(exec));
    as<ConvertibleTo<dense>>(submtx_22->mtx.get())
        ->convert_to(schur_complement.get());

    const auto block_col_ptrs = submtx_21->block_ptrs.get_const_data();
    const auto block_row_ptrs = submtx_12->block_ptrs.get_const_data();
    for (IndexType block = 0; block < num_blocks; block++) {
        std::cout << "block: " << block << '\n';
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        ValueType coeff = -1.0;
        spdgemm_blocks(exec, size, block_size, submtx_11, submtx_12, submtx_21,
                       schur_complement.get(), block, coeff);
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
    const auto block_col_ptrs = submtx_21->block_ptrs.get_const_data();
    const auto block_row_ptrs = submtx_12->block_ptrs.get_const_data();
    const auto values_21 = submtx_21->mtx->get_values();
    const auto row_idxs_21 = submtx_21->mtx->get_col_idxs();
    const auto values_12 = submtx_12->mtx->get_values();
    const auto col_idxs_12 = submtx_12->mtx->get_col_idxs();
    const auto num_rows_21 =
        (block_col_ptrs[block_index + 1] - block_col_ptrs[block_index]) /
        static_cast<IndexType>(block_size);
    const auto num_cols_12 =
        (block_row_ptrs[block_index + 1] - block_row_ptrs[block_index]) /
        static_cast<IndexType>(block_size);
    const auto split_index = submtx_12->split_index;
    auto schur_complement_values = schur_complement->get_values();
    // std::cout << "row_idxs_21[0]: " << row_idxs_21[0] << '\n';
    // std::cout << "row_idxs_21[1]: " << row_idxs_21[1] << '\n';
    // std::cout << "row_idxs_21[2]: " << row_idxs_21[2] << '\n';
    // std::cout << "row_idxs_21[3]: " << row_idxs_21[3] << '\n';
    // std::cout << "row_idxs_21[4]: " << row_idxs_21[4] << '\n';
    // std::cout << "row_idxs_21[5]: " << row_idxs_21[5] << '\n';
    for (auto i = 0; i < block_size; i++) {
        for (auto j = 0; j < num_rows_21; j++) {
            for (auto k = 0; k < num_cols_12; k++) {
                auto col_index_21 =
                    block_col_ptrs[block_index] + num_rows_21 * i + j;
                auto value_21 = values_21[col_index_21];
                auto row = row_idxs_21[col_index_21] -
                           static_cast<IndexType>(split_index);

                auto row_index_12 =
                    block_row_ptrs[block_index] + num_cols_12 * i + k;
                auto value_12 = values_12[row_index_12];
                auto col = col_idxs_12[row_index_12];

                // std::cout << "row: " << row << ", col: " << col << ",
                // col_index_21: " << col_index_21 << '\n';
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL);


// Solve kernels.

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
    const auto num_rows = dense_mtx->get_size()[0];
    auto values_mtx = dense_mtx->get_values();
    for (auto row_local = 0; row_local < size[0]; row_local++) {
        const auto row = row_local + col_start;
        auto row_index = row_ptrs[row];
        auto col_cur = col_idxs[row_index];
        IndexType col_old = -1;
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


}  // namespace arrow_lu
}  // namespace reference
}  // namespace kernels
}  // namespace gko