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

#define PIVOT_AUGMENTATION 1e-8

// Checks rows row_start, ..., row_end of submatrix_12 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <size_type block_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(block_size) void find_min_col_12_kernel(
    IndexType* row_ptrs_current, const IndexType max_row,
    const IndexType* row_ptrs, const IndexType* col_idxs, IndexType row_start,
    IndexType row_end, IndexType* col_min_out, IndexType* row_min_out,
    IndexType* num_occurences_out)
{
    IndexType num_occurences = 0;
    auto col_min = max_row;
    auto row_min = max_row;
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

// Copies all nonzero entries in the wavefront of (1, 2) subblock
// iu global_mtx to submatrix_12.mtx.
template <size_type block_size, typename ValueType, typename IndexType>
__global__
    __launch_bounds__(block_size) void update_row_ptrs_current_in_assemble_12(
        const IndexType* col_idxs_src, const ValueType* values_src,
        IndexType* row_ptrs_current, IndexType* row_ptrs, IndexType* col_idxs,
        ValueType* values, IndexType col_min, IndexType split_index)
{
    auto row = threadIdx.x;
    auto row_index = row_ptrs_current[row];
    auto col = col_idxs_src[row_index];
    // Copies value and column. Updates row_ptr.
    auto row_index_output = row_ptrs[row];
    values[row_index_output] = (col == col_min) ? values_src[row_index] : 0.0;
    col_idxs[row_index_output] = col_min - split_index;
    row_ptrs[row] += 1;
    // Increment wavefront ptr.
    if (col == col_min) {
        row_ptrs_current[row] += 1;
    }
}

template <size_type block_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(block_size) void update_l_factor_row(
    const ValueType* mtx_values, const IndexType num_rows,
    const IndexType current_row, IndexType* l_values)
{
    auto tid = threadIdx.x;
    if (tid == current_row) {
        l_values[num_rows * current_row + current_row] = 1.0;
    } else {
        ValueType pivot = mtx_values[num_rows * current_row + current_row];
        if (abs(pivot) < PIVOT_THRESHOLD) {
            pivot += PIVOT_AUGMENTATION;
        }

        l_values[num_rows * tid + current_row] =
            mtx_values[num_rows * tid + current_row] / pivot;
    }
}

template <size_type block_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(block_size) void update_u_factor_col(
    const ValueType* mtx_values, const IndexType num_rows,
    const IndexType current_row, IndexType* u_values)
{
    auto tid = threadIdx.x;
    ValueType pivot = mtx_values[num_rows * current_row + current_row];
    if (abs(pivot) < PIVOT_THRESHOLD) {
        pivot += PIVOT_AUGMENTATION;
    }

    if (tid == current_row) {
        u_values[num_rows * current_row + current_row] = pivot;
    } else {
        u_values[num_rows * tid + current_row] =
            mtx_values[num_rows * tid + current_row];
    }
}

template <size_type block_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(block_size) void convert_csr_2_dense_kernel(
    const IndexType col_start, const IndexType* row_ptrs,
    const IndexType* col_idxs, const ValueType* values, ValueType* values_mtx,
    const IndexType num_rows, IndexType col_end)
{
    IndexType col_old = -1;
    auto row_local = threadIdx.x;
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

template <typename IndexType>
__device__ void increment_current_row_ptrs(
    IndexType* row_ptrs_current, const IndexType* col_idxs_src,
    const IndexType* row_ptrs_src, IndexType* row_ptrs_submtx_12_current,
    IndexType* col_min, IndexType* remaining_nnz, IndexType row_start,
    IndexType row_end, bool* found_nonzero_column_in)
{
    bool found_nonzero_column = false;
    for (auto row = row_start; row < row_end; row++) {
        auto r_ptr = row_ptrs_current[row];
        auto col = col_idxs_src[r_ptr];
        if ((col == *col_min) &&
            (row_ptrs_current[row] < row_ptrs_src[row + 1])) {
            row_ptrs_current[row] += 1;
            *remaining_nnz -= 1;
            found_nonzero_column = true;
            // Updates number row_ptrs of submatrix_12.
            row_ptrs_submtx_12_current[row] += 1;
        }
    }
    *found_nonzero_column_in = found_nonzero_column;
}

// Checks rows row_start, ..., row_end of submatrix_12 and returns the entry
// (row_min_out, col_min_out) as well as the number of entries with the same
// column index, in variable num_occurences_out.
template <typename IndexType>
__host__ __device__ void find_min_col_kernel(
    IndexType* row_ptrs_current, IndexType max_row, const IndexType* row_ptrs,
    const IndexType* col_idxs, IndexType row_start, IndexType row_end,
    IndexType* col_min_out, IndexType* row_min_out,
    IndexType* num_occurences_out)
{
    IndexType num_occurences = 0;
    auto col_min = max_row;
    auto row_min = max_row;
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

template <typename IndexType>
__device__ IndexType symbolic_count_row_check(IndexType* row_ptrs_current,
                                              const IndexType* row_ptrs_src,
                                              const IndexType row_start,
                                              const IndexType row_end)
{
    auto remaining_nnz = 0;
    for (auto row = row_start; row < row_end; row++) {
        remaining_nnz += ((row_ptrs_src[row + 1] > row_ptrs_current[row])
                              ? row_ptrs_src[row + 1] - row_ptrs_current[row]
                              : 0);
    }
    return remaining_nnz;
}

template <size_type block_size, typename IndexType>
__global__ __launch_bounds__(block_size) void step_1_impl_symbolic_count_kernel(
    IndexType* row_ptrs_current_submtx_12, IndexType* partition_idxs,
    IndexType block, IndexType* row_ptrs_current, const IndexType* row_ptrs_src,
    IndexType* nz_per_block, const IndexType* col_idxs_src,
    IndexType* nnz_submatrix_12_count_in, IndexType split_index,
    const IndexType max_col, IndexType* block_row_ptrs)
{
    // inputs?
    IndexType row_min = 0;
    IndexType col_min = 0;
    IndexType num_occurences = 0;
    IndexType nnz_submatrix_12_count = 0;

    const auto block_length = partition_idxs[block + 1] - partition_idxs[block];
    const auto row_start = partition_idxs[block];
    const auto row_end = partition_idxs[block + 1];
    bool found_nonzero_column;
    // Computes the number of remaining nonzeros in current block of
    // submatrix (1, 2) of global_mtx.
    auto remaining_nnz = symbolic_count_row_check(
        row_ptrs_current, row_ptrs_src, row_start, row_end);
    nz_per_block[block] = remaining_nnz;
    // Computes the number of nonzeros of the column-sparse block and
    // sets up the block_row, row_ptr indices for the current block of
    // submatrix_12.
    while (remaining_nnz > 0) {
        // Finds the (row_min, col_min) entry in submatrix_12 that has the
        // smallest column in the current block.
        find_min_col_kernel(row_ptrs_current_submtx_12, max_col, row_ptrs_src,
                            col_idxs_src, row_start, row_end, &col_min,
                            &row_min, &num_occurences);

        // Increments row_ptrs_current[row] for all entries (row, col) s.t.
        // row in (row_start, row_end) and col =
        // col_idxs[row_ptrs_current[row]].
        increment_current_row_ptrs<IndexType>(
            row_ptrs_current, col_idxs_src, row_ptrs_src,
            row_ptrs_current_submtx_12, &col_min, &remaining_nnz, row_start,
            row_end, &found_nonzero_column);

        // Updates nonzero count, num_cols and col_min
        nnz_submatrix_12_count =
            (found_nonzero_column && (col_min >= split_index))
                ? nnz_submatrix_12_count + block_length
                : nnz_submatrix_12_count;
        col_min = max_col;
    }

    *nnz_submatrix_12_count_in = nnz_submatrix_12_count;

    // Updates the block_row_ptrs array.
    block_row_ptrs[block + 1] =
        nnz_submatrix_12_count;  // this should be done here but there
                                 // should be something in the description
                                 // of this function to detail it.
}