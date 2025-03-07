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

namespace kernel {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool block_segment_scan_reverse(
    const IndexType* __restrict__ ind, ValueType* __restrict__ val)
{
    bool last = true;
    const auto reg_ind = ind[threadIdx.x];
#pragma unroll
    for (int i = 1; i < spmv_block_size; i <<= 1) {
        if (i == 1 && threadIdx.x < spmv_block_size - 1 &&
            reg_ind == ind[threadIdx.x + 1]) {
            last = false;
        }
        auto temp = zero<ValueType>();
        if (threadIdx.x >= i && reg_ind == ind[threadIdx.x - i]) {
            temp = val[threadIdx.x - i];
        }
        group::this_thread_block().sync();
        val[threadIdx.x] += temp;
        group::this_thread_block().sync();
    }

    return last;
}


template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType* __restrict__ row, IndexType* __restrict__ row_end,
    const IndexType row_predict, const IndexType row_predict_end,
    const IndexType* __restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= *row_end) {
            *row = row_predict;
            *row_end = row_predict_end;
            while (ind >= *row_end) {
                *row_end = row_ptr[++*row + 1];
            }
        }

    } else {
        *row = num_rows - 1;
        *row_end = data_size;
    }
}


template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const group::thread_block_tile<subwarp_size>& group, bool force_write,
    ValueType* __restrict__ val, const IndexType row, ValueType* __restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(group, row, val);
    if (need_write && force_write) {
        atomic_add(&(c[row * c_stride + column_id]), scale(*val));
    }
    if (!need_write || force_write) {
        *val = zero<ValueType>();
    }
}


template <bool last, unsigned subwarp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ __forceinline__ void process_window(
    const group::thread_block_tile<subwarp_size>& group,
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType* __restrict__ row, IndexType* __restrict__ row_end,
    IndexType* __restrict__ nrow, IndexType* __restrict__ nrow_end,
    ValueType* __restrict__ temp_val, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    const IndexType curr_row = *row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, *nrow,
                        *nrow_end, row_ptrs);
    // segmented scan
    if (group.any(curr_row != *row)) {
        warp_atomic_add(group, curr_row != *row, temp_val, curr_row, c,
                        c_stride, column_id, scale);
        *nrow = group.shfl(*row, subwarp_size - 1);
        *nrow_end = group.shfl(*row_end, subwarp_size - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        *temp_val += val[ind] * b[col * b_stride + column_id];
    }
}


template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, config::warp_size);
    return (warp_idx * cache_lines / nwarps) * config::warp_size;
}


template <typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride, Closure scale)
{
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    const IndexType column_id = blockIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    constexpr IndexType wsize = config::warp_size;
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    ValueType temp_val = zero<ValueType>();
    IndexType ind = start + threadIdx.x;
    find_next_row<true>(num_rows, data_size, ind, &row, &row_end, nrow,
                        nrow_end, row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block());
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, &row,
                              &row_end, &nrow, &nrow_end, &temp_val, val,
                              col_idxs, row_ptrs, b, b_stride, c, c_stride,
                              column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, &row, &row_end,
                         &nrow, &nrow_end, &temp_val, val, col_idxs, row_ptrs,
                         b, b_stride, c, c_stride, column_id, scale);
    warp_atomic_add(tile_block, true, &temp_val, row, c, c_stride, column_id,
                    scale);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [](const ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [&scale_factor](const ValueType& x) {
                    return scale_factor * x;
                });
}


template <typename IndexType>
__forceinline__ __device__ void merge_path_search(
    const IndexType diagonal, const IndexType a_len, const IndexType b_len,
    const IndexType* __restrict__ a, const IndexType offset_b,
    IndexType* __restrict__ x, IndexType* __restrict__ y)
{
    auto x_min = max(diagonal - b_len, zero<IndexType>());
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        auto pivot = x_min + (x_max - x_min) / 2;
        if (a[pivot] <= offset_b + diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}


template <typename ValueType, typename IndexType, typename Alpha_op>
__device__ void merge_path_reduce(const IndexType nwarps,
                                  const ValueType* __restrict__ last_val,
                                  const IndexType* __restrict__ last_row,
                                  ValueType* __restrict__ c,
                                  const size_type c_stride, Alpha_op alpha_op)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = threadIdx.x;
    const IndexType start = min(tid * cache_lines, nwarps);
    const IndexType end = min((tid + 1) * cache_lines, nwarps);
    ValueType value = zero<ValueType>();
    IndexType row = last_row[nwarps - 1];
    if (start < nwarps) {
        value = last_val[start];
        row = last_row[start];
        for (IndexType i = start + 1; i < end; i++) {
            if (last_row[i] != row) {
                c[row * c_stride] += alpha_op(value);
                row = last_row[i];
                value = last_val[i];
            } else {
                value += last_val[i];
            }
        }
    }
    __shared__ UninitializedArray<IndexType, spmv_block_size> tmp_ind;
    __shared__ UninitializedArray<ValueType, spmv_block_size> tmp_val;
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType*>(tmp_ind),
                                           static_cast<ValueType*>(tmp_val));
    group::this_thread_block().sync();
    if (last) {
        c[row * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}


template <int items_per_thread, typename ValueType, typename IndexType,
          typename Alpha_op, typename Beta_op>
__device__ void merge_path_spmv(
    const IndexType num_rows, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride,
    IndexType* __restrict__ row_out, ValueType* __restrict__ val_out,
    Alpha_op alpha_op, Beta_op beta_op)
{
    const auto* row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const auto block_items = spmv_block_size * items_per_thread;
    __shared__ IndexType shared_row_ptrs[block_items];
    const IndexType diagonal =
        min(IndexType(block_items * blockIdx.x), num_merge_items);
    const IndexType diagonal_end = min(diagonal + block_items, num_merge_items);
    IndexType block_start_x;
    IndexType block_start_y;
    IndexType end_x;
    IndexType end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, zero<IndexType>(),
                      &block_start_x, &block_start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs,
                      zero<IndexType>(), &end_x, &end_y);
    const IndexType block_num_rows = end_x - block_start_x;
    const IndexType block_num_nonzeros = end_y - block_start_y;
    for (int i = threadIdx.x;
         i < block_num_rows && block_start_x + i < num_rows;
         i += spmv_block_size) {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    group::this_thread_block().sync();

    IndexType start_x;
    IndexType start_y;
    merge_path_search(IndexType(items_per_thread * threadIdx.x), block_num_rows,
                      block_num_nonzeros, shared_row_ptrs, block_start_y,
                      &start_x, &start_y);


    IndexType ind = block_start_y + start_y;
    IndexType row_i = block_start_x + start_x;
    ValueType value = zero<ValueType>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (row_i < num_rows) {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x]) {
                value += val[ind] * b[col_idxs[ind] * b_stride];
                ind++;
            } else {
                c[row_i * c_stride] =
                    alpha_op(value) + beta_op(c[row_i * c_stride]);
                start_x++;
                row_i++;
                value = zero<ValueType>();
            }
        }
    }
    group::this_thread_block().sync();
    IndexType* tmp_ind = shared_row_ptrs;
    ValueType* tmp_val =
        reinterpret_cast<ValueType*>(shared_row_ptrs + spmv_block_size);
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row_i;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(tmp_ind, tmp_val);
    if (threadIdx.x == spmv_block_size - 1) {
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    } else if (last) {
        c[row_i * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}

template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride,
    IndexType* __restrict__ row_out, ValueType* __restrict__ val_out)
{
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [](ValueType& x) { return x; },
        [](ValueType& x) { return zero<ValueType>(); });
}


template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    const ValueType* __restrict__ b, const size_type b_stride,
    const ValueType* __restrict__ beta, ValueType* __restrict__ c,
    const size_type c_stride, IndexType* __restrict__ row_out,
    ValueType* __restrict__ val_out)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [&alpha_val](ValueType& x) { return alpha_val * x; },
        [&beta_val](ValueType& x) { return beta_val * x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType* __restrict__ last_val,
    const IndexType* __restrict__ last_row, ValueType* __restrict__ c,
    const size_type c_stride)
{
    merge_path_reduce(nwarps, last_val, last_row, c, c_stride,
                      [](ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType* __restrict__ last_val,
    const IndexType* __restrict__ last_row, const ValueType* __restrict__ alpha,
    ValueType* __restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    merge_path_reduce(nwarps, last_val, last_row, c, c_stride,
                      [&alpha_val](ValueType& x) { return alpha_val * x; });
}


template <size_type subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__device__ void device_classical_spmv(const size_type num_rows,
                                      const ValueType* __restrict__ val,
                                      const IndexType* __restrict__ col_idxs,
                                      const IndexType* __restrict__ row_ptrs,
                                      const ValueType* __restrict__ b,
                                      const size_type b_stride,
                                      ValueType* __restrict__ c,
                                      const size_type c_stride, Closure scale)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subrow = thread::get_subwarp_num_flat<subwarp_size>();
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        ValueType temp_val = zero<ValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += val[ind] * b[col_idxs[ind] * b_stride + column_id];
        }
        auto subwarp_result = reduce(
            subwarp_tile, temp_val,
            [](const ValueType& a, const ValueType& b) { return a + b; });
        if (subid == 0) {
            c[row * c_stride + column_id] =
                scale(subwarp_result, c[row * c_stride + column_id]);
        }
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride)
{
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [](const ValueType& x, const ValueType& y) { return x; });
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ b,
    const size_type b_stride, const ValueType* __restrict__ beta,
    ValueType* __restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
            return alpha_val * x + beta_val * y;
        });
}


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam_nnz(
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const IndexType* __restrict__ b_row_ptrs,
    const IndexType* __restrict__ b_col_idxs, IndexType num_rows,
    IndexType* __restrict__ nnz)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    IndexType count{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType, IndexType a_col, IndexType, IndexType b_col, IndexType,
            bool valid) {
            count += popcnt(subwarp.ballot(a_col != b_col && valid));
            return true;
        });

    if (subwarp.thread_rank() == 0) {
        nnz[row] = count;
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam(
    const ValueType* __restrict__ palpha,
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const ValueType* __restrict__ a_vals, const ValueType* __restrict__ pbeta,
    const IndexType* __restrict__ b_row_ptrs,
    const IndexType* __restrict__ b_col_idxs,
    const ValueType* __restrict__ b_vals, IndexType num_rows,
    const IndexType* __restrict__ c_row_ptrs,
    IndexType* __restrict__ c_col_idxs, ValueType* __restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto alpha = palpha[0];
    const auto beta = pbeta[0];
    const auto lane = static_cast<IndexType>(subwarp.thread_rank());
    constexpr auto lanemask_full =
        ~config::lane_mask_type{} >> (config::warp_size - subwarp_size);
    const auto lanemask_eq = config::lane_mask_type{1} << lane;
    const auto lanemask_lt = lanemask_eq - 1;

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    auto c_begin = c_row_ptrs[row];
    bool skip_first{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType a_nz, IndexType a_col, IndexType b_nz, IndexType b_col,
            IndexType, bool valid) {
            auto c_col = min(a_col, b_col);
            auto equal_mask = subwarp.ballot(a_col == b_col && valid);
            // check if the elements in the previous merge step are
            // equal
            auto prev_equal_mask = equal_mask << 1 | skip_first;
            // store the highest bit for the next group_merge_step
            skip_first = bool(equal_mask >> (subwarp_size - 1));
            auto prev_equal = bool(prev_equal_mask & lanemask_eq);
            // only output an entry if the previous cols weren't equal.
            // if they were equal, they were both handled in the
            // previous step
            if (valid && !prev_equal) {
                auto c_ofs = popcnt(~prev_equal_mask & lanemask_lt);
                c_col_idxs[c_begin + c_ofs] = c_col;
                auto a_val =
                    a_col <= b_col ? a_vals[a_nz + a_begin] : zero<ValueType>();
                auto b_val =
                    b_col <= a_col ? b_vals[b_nz + b_begin] : zero<ValueType>();
                c_vals[c_begin + c_ofs] = alpha * a_val + beta * b_val;
            }
            // advance by the number of merged elements
            // in theory, we would need to mask by `valid`, but this
            // would only be false somwhere in the last iteration, where
            // we don't need the value of c_begin afterwards, anyways.
            c_begin += popcnt(~prev_equal_mask & lanemask_full);
            return true;
        });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, size_type stride,
    ValueType* __restrict__ result)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void extract_diagonal(
    size_type diag_size, size_type nnz,
    const ValueType* __restrict__ orig_values,
    const IndexType* __restrict__ orig_row_ptrs,
    const IndexType* __restrict__ orig_col_idxs, ValueType* __restrict__ diag)
{
    constexpr auto warp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<warp_size>();
    const auto local_tidx = threadIdx.x % warp_size;

    if (row < diag_size) {
        for (size_type i = local_tidx;
             i < orig_row_ptrs[row + 1] - orig_row_ptrs[row]; i += warp_size) {
            const auto orig_idx = i + orig_row_ptrs[row];
            if (orig_idx < nnz) {
                if (orig_col_idxs[orig_idx] == row) {
                    diag[row] = orig_values[orig_idx];
                    return;
                }
            }
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void row_ptr_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs, IndexType* __restrict__ out_nnz)
{
    auto tid = thread::get_thread_id_flat();
    if (tid >= num_rows) {
        return;
    }
    auto in_row = permutation[tid];
    auto out_row = tid;
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_row_ptr_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs, IndexType* __restrict__ out_nnz)
{
    auto tid = thread::get_thread_id_flat();
    if (tid >= num_rows) {
        return;
    }
    auto in_row = tid;
    auto out_row = permutation[tid];
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void row_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    auto lane = threadIdx.x % subwarp_size;
    auto in_row = permutation[tid];
    auto out_row = tid;
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_row_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    auto lane = threadIdx.x % subwarp_size;
    auto in_row = tid;
    auto out_row = permutation[tid];
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_symm_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    auto lane = threadIdx.x % subwarp_size;
    auto in_row = tid;
    auto out_row = permutation[tid];
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void compute_submatrix_idxs_and_vals(
        const size_type num_rows, const size_type num_cols,
        const size_type row_offset, const size_type col_offset,
        const IndexType* __restrict__ source_row_ptrs,
        const IndexType* __restrict__ source_col_idxs,
        const ValueType* __restrict__ source_values,
        const IndexType* __restrict__ result_row_ptrs,
        IndexType* __restrict__ result_col_idxs,
        ValueType* __restrict__ result_values)
{
    const auto res_row = thread::get_thread_id_flat();
    if (res_row < num_rows) {
        const auto src_row = res_row + row_offset;
        auto res_nnz = result_row_ptrs[res_row];
        for (auto nnz = source_row_ptrs[src_row];
             nnz < source_row_ptrs[src_row + 1]; ++nnz) {
            const auto res_col =
                source_col_idxs[nnz] - static_cast<IndexType>(col_offset);
            if (res_col < num_cols && res_col >= 0) {
                result_col_idxs[res_nnz] = res_col;
                result_values[res_nnz] = source_values[nnz];
                res_nnz++;
            }
        }
    }
}


template <typename IndexType>
__global__
    __launch_bounds__(default_block_size) void calculate_nnz_per_row_in_span(
        const span row_span, const span col_span,
        const IndexType* __restrict__ row_ptrs,
        const IndexType* __restrict__ col_idxs,
        IndexType* __restrict__ nnz_per_row)
{
    const auto src_row = thread::get_thread_id_flat() + row_span.begin;
    if (src_row < row_span.end) {
        IndexType nnz{};
        for (auto i = row_ptrs[src_row]; i < row_ptrs[src_row + 1]; ++i) {
            if (col_idxs[i] >= col_span.begin && col_idxs[i] < col_span.end) {
                nnz++;
            }
        }
        nnz_per_row[src_row - row_span.begin] = nnz;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_scaled_identity(
    const ValueType* const __restrict__ alpha,
    const ValueType* const __restrict__ beta, const IndexType num_rows,
    const IndexType* const __restrict__ row_ptrs,
    const IndexType* const __restrict__ col_idxs,
    ValueType* const __restrict__ values)
{
    constexpr int warp_size = config::warp_size;
    auto tile_grp =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto warpid = thread::get_subwarp_id_flat<warp_size, IndexType>();
    const auto num_warps = thread::get_subwarp_num_flat<warp_size, IndexType>();
    if (warpid < num_rows) {
        const auto tid_in_warp = tile_grp.thread_rank();
        const IndexType row_start = row_ptrs[warpid];
        const IndexType num_nz = row_ptrs[warpid + 1] - row_start;
        for (IndexType iz = tid_in_warp; iz < num_nz; iz += warp_size) {
            values[iz + row_start] *= beta[0];
            if (col_idxs[iz + row_start] == warpid) {
                values[iz + row_start] += alpha[0];
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source, int bs,
                      array<IndexType>& block_row_ptr_array,
                      array<IndexType>& block_col_idx_array,
                      array<ValueType>& block_value_array)
{
    using tuple_type = thrust::tuple<IndexType, IndexType>;
    const auto nnz = source->get_num_stored_elements();
    array<IndexType> in_row_idxs{exec, nnz};
    array<IndexType> in_col_idxs{exec, nnz};
    array<ValueType> in_values{exec, nnz};
    exec->copy(nnz, source->get_const_col_idxs(), in_col_idxs.get_data());
    exec->copy(nnz, source->get_const_values(), in_values.get_data());
    components::convert_ptrs_to_idxs(exec, source->get_const_row_ptrs(),
                                     source->get_size()[0],
                                     in_row_idxs.get_data());
    auto block_row_ptrs = block_row_ptr_array.get_data();
    auto num_block_rows = block_row_ptr_array.get_num_elems() - 1;
    if (nnz == 0) {
        components::fill_array(exec, block_row_ptrs, num_block_rows + 1,
                               IndexType{});
        block_col_idx_array.resize_and_reset(0);
        block_value_array.resize_and_reset(0);
        return;
    }
    auto in_rows = in_row_idxs.get_data();
    auto in_cols = in_col_idxs.get_data();
    // workaround for CUDA 9.2 Thrust: Their complex<> implementation is broken
    // due to overly generic assignment operator and constructor leading to
    // ambiguities. So we need to use our own fake_complex type
    auto in_vals =
        reinterpret_cast<device_member_type<ValueType>*>(in_values.get_data());
    auto in_loc_it = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(in_rows),
                           thrust::device_pointer_cast(in_cols)));
    thrust::sort_by_key(thrust::device, in_loc_it, in_loc_it + nnz,
                        thrust::device_pointer_cast(in_vals),
                        [bs] __device__(tuple_type a, tuple_type b) {
                            return thrust::make_pair(thrust::get<0>(a) / bs,
                                                     thrust::get<1>(a) / bs) <
                                   thrust::make_pair(thrust::get<0>(b) / bs,
                                                     thrust::get<1>(b) / bs);
                        });
    // build block pattern
    auto adj_predicate = [bs, in_rows, in_cols, nnz] __device__(size_type i) {
        const auto a_block_row = i > 0 ? in_rows[i - 1] / bs : -1;
        const auto a_block_col = i > 0 ? in_cols[i - 1] / bs : -1;
        const auto b_block_row = in_rows[i] / bs;
        const auto b_block_col = in_cols[i] / bs;
        return (a_block_row != b_block_row) || (a_block_col != b_block_col);
    };
    auto iota = thrust::make_counting_iterator(size_type{});
    // count how many blocks we have by counting how often the block changes
    auto num_blocks = static_cast<size_type>(
        thrust::count_if(thrust::device, iota, iota + nnz, adj_predicate));
    // allocate storage
    array<IndexType> block_row_idx_array{exec, num_blocks};
    array<size_type> block_ptr_array{exec, num_blocks};
    block_col_idx_array.resize_and_reset(num_blocks);
    block_value_array.resize_and_reset(num_blocks * bs * bs);
    auto row_idxs = block_row_idx_array.get_data();
    auto col_idxs = block_col_idx_array.get_data();
    auto values = as_device_type(block_value_array.get_data());
    auto block_ptrs = block_ptr_array.get_data();
    auto block_ptr_it = thrust::device_pointer_cast(block_ptrs);
    // write (block_row, block_col, block_start_idx) tuples for each block
    thrust::copy_if(thrust::device, iota, iota + nnz, block_ptr_it,
                    adj_predicate);
    auto block_output_it = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(row_idxs),
                           thrust::device_pointer_cast(col_idxs)));
    thrust::transform(
        thrust::device, block_ptr_it, block_ptr_it + num_blocks,
        block_output_it, [bs, in_rows, in_cols] __device__(size_type i) {
            return thrust::make_tuple(in_rows[i] / bs, in_cols[i] / bs);
        });
    // build row pointers from row indices
    components::convert_idxs_to_ptrs(exec, block_row_idx_array.get_const_data(),
                                     block_row_idx_array.get_num_elems(),
                                     num_block_rows, block_row_ptrs);
    // fill in values
    components::fill_array(exec, block_value_array.get_data(),
                           num_blocks * bs * bs, zero<ValueType>());
    thrust::for_each_n(
        thrust::device, iota, num_blocks,
        [block_ptrs, nnz, num_blocks, bs, in_rows, in_cols, in_vals,
         values] __device__(size_type i) {
            const auto block_begin = block_ptrs[i];
            const auto block_end = i < num_blocks - 1 ? block_ptrs[i + 1] : nnz;
            for (auto nz = block_begin; nz < block_end; nz++) {
                values[i * bs * bs + (in_cols[nz] % bs) * bs +
                       (in_rows[nz] % bs)] = fake_complex_unpack(in_vals[nz]);
            }
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);
