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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/base/allocator.hpp"
#include "core/base/index_set_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "reference/components/csr_spgeam.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Csr
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Csr<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Csr<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) *= vbeta;
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += valpha * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm_insert_row(unordered_set<IndexType>& cols,
                       const matrix::Csr<ValueType, IndexType>* c,
                       size_type row)
{
    auto row_ptrs = c->get_const_row_ptrs();
    auto col_idxs = c->get_const_col_idxs();
    cols.insert(col_idxs + row_ptrs[row], col_idxs + row_ptrs[row + 1]);
}


template <typename ValueType, typename IndexType>
void spgemm_insert_row2(unordered_set<IndexType>& cols,
                        const matrix::Csr<ValueType, IndexType>* a,
                        const matrix::Csr<ValueType, IndexType>* b,
                        size_type row)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    for (size_type a_nz = a_row_ptrs[row];
         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
        auto a_col = a_col_idxs[a_nz];
        auto b_row = a_col;
        cols.insert(b_col_idxs + b_row_ptrs[b_row],
                    b_col_idxs + b_row_ptrs[b_row + 1]);
    }
}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row(map<IndexType, ValueType>& cols,
                           const matrix::Csr<ValueType, IndexType>* c,
                           ValueType scale, size_type row)
{
    auto row_ptrs = c->get_const_row_ptrs();
    auto col_idxs = c->get_const_col_idxs();
    auto vals = c->get_const_values();
    for (size_type c_nz = row_ptrs[row]; c_nz < size_type(row_ptrs[row + 1]);
         ++c_nz) {
        auto c_col = col_idxs[c_nz];
        auto c_val = vals[c_nz];
        cols[c_col] += scale * c_val;
    }
}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row2(map<IndexType, ValueType>& cols,
                            const matrix::Csr<ValueType, IndexType>* a,
                            const matrix::Csr<ValueType, IndexType>* b,
                            ValueType scale, size_type row)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    for (size_type a_nz = a_row_ptrs[row];
         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
        auto a_col = a_col_idxs[a_nz];
        auto a_val = a_vals[a_nz];
        auto b_row = a_col;
        for (size_type b_nz = b_row_ptrs[b_row];
             b_nz < size_type(b_row_ptrs[b_row + 1]); ++b_nz) {
            auto b_col = b_col_idxs[b_nz];
            auto b_val = b_vals[b_nz];
            cols[b_col] += scale * a_val * b_val;
        }
    }
}


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    auto num_rows = a->get_size()[0];

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    unordered_set<IndexType> local_col_idxs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_col_idxs.clear();
        spgemm_insert_row2(local_col_idxs, a, b, a_row);
        c_row_ptrs[a_row] = static_cast<IndexType>(local_col_idxs.size());
    }

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    map<IndexType, ValueType> local_row_nzs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_row_nzs.clear();
        spgemm_accumulate_row2(local_row_nzs, a, b, one<ValueType>(), a_row);
        // store result
        auto c_nz = c_row_ptrs[a_row];
        for (auto pair : local_row_nzs) {
            c_col_idxs[c_nz] = pair.first;
            c_vals[c_nz] = pair.second;
            ++c_nz;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     const matrix::Dense<ValueType>* beta,
                     const matrix::Csr<ValueType, IndexType>* d,
                     matrix::Csr<ValueType, IndexType>* c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    unordered_set<IndexType> local_col_idxs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_col_idxs.clear();
        spgemm_insert_row(local_col_idxs, d, a_row);
        spgemm_insert_row2(local_col_idxs, a, b, a_row);
        c_row_ptrs[a_row] = static_cast<IndexType>(local_col_idxs.size());
    }

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    map<IndexType, ValueType> local_row_nzs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_row_nzs.clear();
        spgemm_accumulate_row(local_row_nzs, d, vbeta, a_row);
        spgemm_accumulate_row2(local_row_nzs, a, b, valpha, a_row);
        // store result
        auto c_nz = c_row_ptrs[a_row];
        for (auto pair : local_row_nzs) {
            c_col_idxs[c_nz] = pair.first;
            c_vals[c_nz] = pair.second;
            ++c_nz;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType>* alpha,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Dense<ValueType>* beta,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    abstract_spgeam(
        a, b, [](IndexType) { return IndexType{}; },
        [](IndexType, IndexType, ValueType, ValueType, IndexType& nnz) {
            ++nnz;
        },
        [&](IndexType row, IndexType nnz) { c_row_ptrs[row] = nnz; });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    abstract_spgeam(
        a, b, [&](IndexType row) { return c_row_ptrs[row]; },
        [&](IndexType, IndexType col, ValueType a_val, ValueType b_val,
            IndexType& nz) {
            c_vals[nz] = valpha * a_val + vbeta * b_val;
            c_col_idxs[nz] = col;
            ++nz;
        },
        [](IndexType, IndexType) {});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Csr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto vals = source->get_const_values();

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = row_ptrs[row];
             i < static_cast<size_type>(row_ptrs[row + 1]); ++i) {
            result->at(row, col_idxs[i]) = vals[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source,
                      matrix::Sellp<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = result->get_slice_size();
    auto stride_factor = result->get_stride_factor();

    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_values = source->get_const_values();

    auto slice_num = ceildiv(num_rows, slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            size_type sellp_ind = slice_sets[slice] * slice_size + row;
            for (size_type csr_ind = source_row_ptrs[global_row];
                 csr_ind < source_row_ptrs[global_row + 1]; csr_ind++) {
                vals[sellp_ind] = source_values[csr_ind];
                col_idxs[sellp_ind] = source_col_idxs[csr_ind];
                sellp_ind += slice_size;
            }
            for (size_type i = sellp_ind;
                 i <
                 (slice_sets[slice] + slice_lengths[slice]) * slice_size + row;
                 i += slice_size) {
                col_idxs[i] = invalid_index<IndexType>();
                vals[i] = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* source,
                    matrix::Ell<ValueType, IndexType>* result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto vals = source->get_const_values();
    const auto col_idxs = source->get_const_col_idxs();
    const auto row_ptrs = source->get_const_row_ptrs();

    const auto num_stored_elements_per_row =
        result->get_num_stored_elements_per_row();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            result->val_at(row, i) = zero<ValueType>();
            result->col_at(row, i) = invalid_index<IndexType>();
        }
        for (size_type col_idx = 0; col_idx < row_ptrs[row + 1] - row_ptrs[row];
             col_idx++) {
            result->val_at(row, col_idx) = vals[row_ptrs[row] + col_idx];
            result->col_at(row, col_idx) = col_idxs[row_ptrs[row] + col_idx];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source, int bs,
                      array<IndexType>& row_ptrs, array<IndexType>& col_idxs,
                      array<ValueType>& values)
{
    using entry = matrix_data_entry<ValueType, IndexType>;
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto num_block_rows = num_rows / bs;
    const auto num_block_cols = num_cols / bs;
    const auto in_row_ptrs = source->get_const_row_ptrs();
    const auto in_cols = source->get_const_col_idxs();
    const auto in_vals = source->get_const_values();
    const auto nnz = source->get_num_stored_elements();
    auto out_row_ptrs = row_ptrs.get_data();
    array<entry> entry_array{exec, nnz};
    auto entries = entry_array.get_data();
    for (IndexType row = 0; row < num_rows; row++) {
        for (auto nz = in_row_ptrs[row]; nz < in_row_ptrs[row + 1]; nz++) {
            entries[nz] = {row, in_cols[nz], in_vals[nz]};
        }
    }
    auto to_block = [bs](entry a) {
        return std::make_pair(a.row / bs, a.column / bs);
    };
    // sort by block in row-major order
    std::sort(entries, entries + nnz,
              [&](entry a, entry b) { return to_block(a) < to_block(b); });
    // set row pointers by jumps in block row index
    gko::vector<IndexType> col_idx_vec{{exec}};
    gko::vector<ValueType> value_vec{{exec}};
    int64 block_row = -1;
    int64 block_col = -1;
    for (size_type i = 0; i < nnz; i++) {
        const auto entry = entries[i];
        const auto new_block_row = entry.row / bs;
        const auto new_block_col = entry.column / bs;
        while (new_block_row > block_row) {
            // we finished row block_row, so store its end pointer
            out_row_ptrs[block_row + 1] = col_idx_vec.size();
            block_col = -1;
            ++block_row;
        }
        if (new_block_col != block_col) {
            // we encountered a new column, so insert it with block storage
            col_idx_vec.emplace_back(new_block_col);
            value_vec.resize(value_vec.size() + bs * bs);
            block_col = new_block_col;
        }
        const auto local_row = entry.row % bs;
        const auto local_col = entry.column % bs;
        value_vec[value_vec.size() - bs * bs + local_row + local_col * bs] =
            entry.value;
    }
    while (block_row < static_cast<int64>(row_ptrs.get_num_elems() - 1)) {
        // we finished row block_row, so store its end pointer
        out_row_ptrs[block_row + 1] = col_idx_vec.size();
        ++block_row;
    }
    values.resize_and_reset(value_vec.size());
    col_idxs.resize_and_reset(col_idx_vec.size());
    std::copy(value_vec.begin(), value_vec.end(), values.get_data());
    std::copy(col_idx_vec.begin(), col_idx_vec.end(), col_idxs.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_csr_to_csc(size_type num_rows, const IndexType* row_ptrs,
                               const IndexType* col_idxs,
                               const ValueType* csr_vals, IndexType* row_idxs,
                               IndexType* col_ptrs, ValueType* csc_vals,
                               UnaryOperator op)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = row;
            csc_vals[dest_idx] = op(csr_vals[i]);
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const ReferenceExecutor> exec,
                             matrix::Csr<ValueType, IndexType>* trans,
                             const matrix::Csr<ValueType, IndexType>* orig,
                             UnaryOperator op)
{
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto trans_vals = trans->get_values();
    auto orig_vals = orig->get_const_values();

    auto orig_num_cols = orig->get_size()[1];
    auto orig_num_rows = orig->get_size()[0];
    auto orig_nnz = orig_row_ptrs[orig_num_rows];

    components::fill_array(exec, trans_row_ptrs, orig_num_cols + 1,
                           IndexType{});
    for (size_type i = 0; i < orig_nnz; i++) {
        trans_row_ptrs[orig_col_idxs[i] + 1]++;
    }
    components::prefix_sum(exec, trans_row_ptrs + 1, orig_num_cols);

    convert_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs, orig_vals,
                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* orig,
               matrix::Csr<ValueType, IndexType>* trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* orig,
                    matrix::Csr<ValueType, IndexType>* trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_span(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source, const span& row_span,
    const span& col_span, array<IndexType>* row_nnz)
{
    size_type res_row = 0;
    for (auto row = row_span.begin; row < row_span.end; ++row) {
        row_nnz->get_data()[res_row] = zero<IndexType>();
        for (auto nnz = source->get_const_row_ptrs()[row];
             nnz < source->get_const_row_ptrs()[row + 1]; ++nnz) {
            if (source->get_const_col_idxs()[nnz] < col_span.end &&
                source->get_const_col_idxs()[nnz] >= col_span.begin) {
                row_nnz->get_data()[res_row]++;
            }
        }
        res_row++;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set, IndexType* row_nnz)
{
    auto num_row_subsets = row_index_set.get_num_subsets();
    auto row_subset_begin = row_index_set.get_subsets_begin();
    auto row_subset_end = row_index_set.get_subsets_end();
    auto row_superset_indices = row_index_set.get_superset_indices();
    auto num_col_subsets = col_index_set.get_num_subsets();
    auto col_subset_begin = col_index_set.get_subsets_begin();
    auto col_subset_end = col_index_set.get_subsets_end();
    auto src_ptrs = source->get_const_row_ptrs();
    for (size_type set = 0; set < num_row_subsets; ++set) {
        size_type res_row = row_superset_indices[set];
        for (auto row = row_subset_begin[set]; row < row_subset_end[set];
             ++row) {
            row_nnz[res_row] = zero<IndexType>();
            for (size_type i = src_ptrs[row]; i < src_ptrs[row + 1]; ++i) {
                auto index = source->get_const_col_idxs()[i];
                if (index >= col_index_set.get_size()) {
                    continue;
                }
                const auto bucket = std::distance(
                    col_subset_begin,
                    std::upper_bound(col_subset_begin,
                                     col_subset_begin + num_col_subsets,
                                     index));
                auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
                if (col_subset_end[shifted_bucket] <= index ||
                    (index < col_subset_begin[shifted_bucket])) {
                    continue;
                } else {
                    row_nnz[res_row]++;
                }
            }
            res_row++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       gko::span row_span, gko::span col_span,
                       matrix::Csr<ValueType, IndexType>* result)
{
    auto row_offset = row_span.begin;
    auto col_offset = col_span.begin;
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto res_row_ptrs = result->get_row_ptrs();
    auto res_col_idxs = result->get_col_idxs();
    auto res_values = result->get_values();
    const auto src_row_ptrs = source->get_const_row_ptrs();
    const auto src_col_idxs = source->get_const_col_idxs();
    const auto src_values = source->get_const_values();

    size_type res_nnz = 0;
    for (size_type nnz = 0; nnz < source->get_num_stored_elements(); ++nnz) {
        if (nnz >= src_row_ptrs[row_offset] &&
            nnz < src_row_ptrs[row_offset + num_rows] &&
            (src_col_idxs[nnz] < (col_offset + num_cols) &&
             src_col_idxs[nnz] >= col_offset)) {
            res_col_idxs[res_nnz] = src_col_idxs[nnz] - col_offset;
            res_values[res_nnz] = src_values[nnz];
            res_nnz++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix_from_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_row_subsets = row_index_set.get_num_subsets();
    auto row_subset_begin = row_index_set.get_subsets_begin();
    auto row_subset_end = row_index_set.get_subsets_end();
    auto res_row_ptrs = result->get_row_ptrs();
    auto res_col_idxs = result->get_col_idxs();
    auto res_values = result->get_values();
    auto num_col_subsets = col_index_set.get_num_subsets();
    auto col_subset_begin = col_index_set.get_subsets_begin();
    auto col_subset_end = col_index_set.get_subsets_end();
    auto col_superset_indices = col_index_set.get_superset_indices();
    const auto src_ptrs = source->get_const_row_ptrs();
    const auto src_col_idxs = source->get_const_col_idxs();
    const auto src_values = source->get_const_values();

    size_type res_nnz = 0;
    for (size_type set = 0; set < num_row_subsets; ++set) {
        for (auto row = row_subset_begin[set]; row < row_subset_end[set];
             ++row) {
            for (size_type i = src_ptrs[row]; i < src_ptrs[row + 1]; ++i) {
                auto index = source->get_const_col_idxs()[i];
                if (index >= col_index_set.get_size()) {
                    continue;
                }
                const auto bucket = std::distance(
                    col_subset_begin,
                    std::upper_bound(col_subset_begin,
                                     col_subset_begin + num_col_subsets,
                                     index));
                auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
                if (col_subset_end[shifted_bucket] <= index ||
                    (index < col_subset_begin[shifted_bucket])) {
                    continue;
                } else {
                    res_col_idxs[res_nnz] =
                        index - col_subset_begin[shifted_bucket] +
                        col_superset_indices[shifted_bucket];
                    res_values[res_nnz] = src_values[i];
                    res_nnz++;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       const int64*,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = result->get_ell_num_stored_elements_per_row();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();

    // Initial Hybrid Matrix
    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
         i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = invalid_index<IndexType>();
        }
    }

    const auto csr_row_ptrs = source->get_const_row_ptrs();
    const auto csr_vals = source->get_const_values();
    size_type csr_idx = 0;
    size_type coo_idx = 0;
    for (IndexType row = 0; row < num_rows; row++) {
        size_type ell_idx = 0;
        while (csr_idx < csr_row_ptrs[row + 1]) {
            const auto val = csr_vals[csr_idx];
            if (ell_idx < ell_lim) {
                result->ell_val_at(row, ell_idx) = val;
                result->ell_col_at(row, ell_idx) =
                    source->get_const_col_idxs()[csr_idx];
                ell_idx++;
            } else {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = source->get_const_col_idxs()[csr_idx];
                coo_row[coo_idx] = row;
                coo_idx++;
            }
            csr_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename IndexType>
void invert_permutation(std::shared_ptr<const DefaultExecutor> exec,
                        size_type size, const IndexType* permutation_indices,
                        IndexType* inv_permutation)
{
    for (IndexType i = 0; i < static_cast<IndexType>(size); ++i) {
        inv_permutation[permutation_indices[i]] = i;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                      const IndexType* perm,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Csr<ValueType, IndexType>* permuted)
{
    auto in_row_ptrs = orig->get_const_row_ptrs();
    auto in_col_idxs = orig->get_const_col_idxs();
    auto in_vals = orig->get_const_values();
    auto p_row_ptrs = permuted->get_row_ptrs();
    auto p_col_idxs = permuted->get_col_idxs();
    auto p_vals = permuted->get_values();
    size_type num_rows = orig->get_size()[0];

    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        p_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum(exec, p_row_ptrs, num_rows + 1);
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = p_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        for (IndexType i = 0; i < row_size; ++i) {
            p_col_idxs[dst_begin + i] = perm[in_col_idxs[src_begin + i]];
            p_vals[dst_begin + i] = in_vals[src_begin + i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const IndexType* perm,
                 const matrix::Csr<ValueType, IndexType>* orig,
                 matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto in_row_ptrs = orig->get_const_row_ptrs();
    auto in_col_idxs = orig->get_const_col_idxs();
    auto in_vals = orig->get_const_values();
    auto rp_row_ptrs = row_permuted->get_row_ptrs();
    auto rp_col_idxs = row_permuted->get_col_idxs();
    auto rp_vals = row_permuted->get_values();
    size_type num_rows = orig->get_size()[0];

    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = perm[row];
        auto dst_row = row;
        rp_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum(exec, rp_row_ptrs, num_rows + 1);
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = perm[row];
        auto dst_row = row;
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(in_col_idxs + src_begin, row_size, rp_col_idxs + dst_begin);
        std::copy_n(in_vals + src_begin, row_size, rp_vals + dst_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                         const IndexType* perm,
                         const matrix::Csr<ValueType, IndexType>* orig,
                         matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto in_row_ptrs = orig->get_const_row_ptrs();
    auto in_col_idxs = orig->get_const_col_idxs();
    auto in_vals = orig->get_const_values();
    auto rp_row_ptrs = row_permuted->get_row_ptrs();
    auto rp_col_idxs = row_permuted->get_col_idxs();
    auto rp_vals = row_permuted->get_values();
    size_type num_rows = orig->get_size()[0];

    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        rp_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum(exec, rp_row_ptrs, num_rows + 1);
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(in_col_idxs + src_begin, row_size, rp_col_idxs + dst_begin);
        std::copy_n(in_vals + src_begin, row_size, rp_vals + dst_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const ReferenceExecutor> exec,
                            const IndexType* perm,
                            const matrix::Csr<ValueType, IndexType>* orig,
                            matrix::Csr<ValueType, IndexType>* column_permuted)
{
    auto in_row_ptrs = orig->get_const_row_ptrs();
    auto in_col_idxs = orig->get_const_col_idxs();
    auto in_vals = orig->get_const_values();
    auto cp_row_ptrs = column_permuted->get_row_ptrs();
    auto cp_col_idxs = column_permuted->get_col_idxs();
    auto cp_vals = column_permuted->get_values();
    auto num_rows = orig->get_size()[0];

    for (size_type row = 0; row < num_rows; ++row) {
        auto row_begin = in_row_ptrs[row];
        auto row_end = in_row_ptrs[row + 1];
        cp_row_ptrs[row] = in_row_ptrs[row];
        for (auto k = row_begin; k < row_end; ++k) {
            cp_col_idxs[k] = perm[in_col_idxs[k]];
            cp_vals[k] = in_vals[k];
        }
    }
    cp_row_ptrs[num_rows] = in_row_ptrs[num_rows];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Csr<ValueType, IndexType>* to_sort)
{
    auto values = to_sort->get_values();
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        auto it = detail::make_zip_iterator(col_idxs + start_row_idx,
                                            values + start_row_idx);
        std::sort(it, it + row_nnz, [](auto t1, auto t2) {
            return std::get<0>(t1) < std::get<0>(t2);
        });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* to_check, bool* is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    for (size_type i = 0; i < size[0]; ++i) {
        for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
            if (col_idxs[idx - 1] > col_idxs[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const auto diag_size = diag->get_size()[0];
    auto diag_values = diag->get_values();

    for (size_type row = 0; row < diag_size; ++row) {
        for (size_type idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] == row) {
                diag_values[row] = values[idx];
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


template <typename ValueType, typename IndexType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           matrix::Csr<ValueType, IndexType>* to_scale)
{
    const auto nnz = to_scale->get_num_stored_elements();
    auto values = to_scale->get_values();

    for (size_type idx = 0; idx < nnz; idx++) {
        values[idx] *= alpha->at(0, 0);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_scale(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Dense<ValueType>* alpha,
               matrix::Csr<ValueType, IndexType>* to_scale)
{
    const auto nnz = to_scale->get_num_stored_elements();
    auto values = to_scale->get_values();

    for (size_type idx = 0; idx < nnz; idx++) {
        values[idx] /= alpha->at(0, 0);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const mtx, bool& has_all_diags)
{
    has_all_diags = true;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const size_type minsize = std::min(mtx->get_size()[0], mtx->get_size()[1]);
    for (size_type row = 0; row < minsize; row++) {
        bool row_diag = false;
        for (IndexType iz = row_ptrs[row]; iz < row_ptrs[row + 1]; iz++) {
            if (col_idxs[iz] == row) {
                row_diag = true;
            }
        }
        if (!row_diag) {
            has_all_diags = false;
            break;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const ReferenceExecutor> exec,
                         const matrix::Dense<ValueType>* const alpha,
                         const matrix::Dense<ValueType>* const beta,
                         matrix::Csr<ValueType, IndexType>* const mtx)
{
    const auto nrows = static_cast<IndexType>(mtx->get_size()[0]);
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto vals = mtx->get_values();
    for (IndexType row = 0; row < nrows; row++) {
        for (IndexType iz = row_ptrs[row]; iz < row_ptrs[row + 1]; iz++) {
            vals[iz] *= beta->get_const_values()[0];
            if (row == mtx->get_const_col_idxs()[iz]) {
                vals[iz] += alpha->get_const_values()[0];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
