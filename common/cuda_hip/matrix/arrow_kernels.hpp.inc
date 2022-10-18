/*******************************<GINKGO
LICENSE>****************************** Copyright (c) 2017-2022, the Ginkgo
authors All rights reserved.

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
******************************<GINKGO
LICENSE>*******************************/

#include "core/matrix/arrow_kernels.hpp"


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
#include "core/matrix/arrow_builder.hpp"
// #include "reference/components/arrow_spgeam.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Arrow sparse matrix format namespace.
 * @ref Arrow
 * @ingroup arrow
 */
namespace arrow {


// template <typename ValueType, typename IndexType>
// void spmv(std::shared_ptr<const ReferenceExecutor> exec,
//           const matrix::Arrow<ValueType, IndexType>* a,
//           const matrix::Dense<ValueType>* b,
//           matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROW_SPMV_KERNEL);


// template <typename ValueType, typename IndexType>
// void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
//                    const matrix::Dense<ValueType>* alpha,
//                    const matrix::Arrow<ValueType, IndexType>* a,
//                    const matrix::Dense<ValueType>* b,
//                    const matrix::Dense<ValueType>* beta,
//                    matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROW_ADVANCED_SPMV_KERNEL);


// template <typename ValueType, typename IndexType>
// void spgemm_insert_row(unordered_set<IndexType>& cols,
//                        const matrix::Arrow<ValueType, IndexType>* c,
//                        size_type row) GKO_NOT_IMPLEMENTED;


// template <typename ValueType, typename IndexType>
// void spgemm_insert_row2(unordered_set<IndexType>& cols,
//                         const matrix::Arrow<ValueType, IndexType>* a,
//                         const matrix::Arrow<ValueType, IndexType>* b,
//                         size_type row) GKO_NOT_IMPLEMENTED;

// template <typename ValueType, typename IndexType>
// void spgemm_accumulate_row(map<IndexType, ValueType>& cols,
//                            const matrix::Arrow<ValueType, IndexType>* c,
//                            ValueType scale, size_type row)
//                            GKO_NOT_IMPLEMENTED;


// template <typename ValueType, typename IndexType>
// void spgemm_accumulate_row2(map<IndexType, ValueType>& cols,
//                             const matrix::Arrow<ValueType, IndexType>* a,
//                             const matrix::Arrow<ValueType, IndexType>* b,
//                             ValueType scale, size_type row)
//                             GKO_NOT_IMPLEMENTED;


// template <typename ValueType, typename IndexType>
// void spgemm(std::shared_ptr<const ReferenceExecutor> exec,
//             const matrix::Arrow<ValueType, IndexType>* a,
//             const matrix::Arrow<ValueType, IndexType>* b,
//             matrix::Arrow<ValueType, IndexType>* c) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


// template <typename ValueType, typename IndexType>
// void advanced_spgemm(std::shared_ptr<const ReferenceExecutor> exec,
//                      const matrix::Dense<ValueType>* alpha,
//                      const matrix::Arrow<ValueType, IndexType>* a,
//                      const matrix::Arrow<ValueType, IndexType>* b,
//                      const matrix::Dense<ValueType>* beta,
//                      const matrix::Arrow<ValueType, IndexType>* d,
//                      matrix::Arrow<ValueType, IndexType>* c)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


// template <typename ValueType, typename IndexType>
// void spgeam(std::shared_ptr<const ReferenceExecutor> exec,
//             const matrix::Dense<ValueType>* alpha,
//             const matrix::Arrow<ValueType, IndexType>* a,
//             const matrix::Dense<ValueType>* beta,
//             const matrix::Arrow<ValueType, IndexType>* b,
//             matrix::Arrow<ValueType, IndexType>* c) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


// template <typename ValueType, typename IndexType>
// void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
//                    const matrix::Arrow<ValueType, IndexType>* source,
//                    matrix::Dense<ValueType>* result) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL);


// template <typename ValueType, typename IndexType>
// void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
//                       const matrix::Arrow<ValueType, IndexType>* source,
//                       matrix::Sellp<ValueType, IndexType>* result)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


// template <typename ValueType, typename IndexType>
// void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
//                     const matrix::Arrow<ValueType, IndexType>* source,
//                     matrix::Ell<ValueType, IndexType>* result)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


// template <typename ValueType, typename IndexType>
// void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
//                       const matrix::Arrow<ValueType, IndexType>* source, int
//                       bs, array<IndexType>& row_ptrs, array<IndexType>&
//                       col_idxs, array<ValueType>& values)
//                       GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);


// template <typename ValueType, typename IndexType, typename UnaryOperator>
// inline void convert_arrow_to_csc(size_type num_rows, const IndexType*
// row_ptrs,
//                                  const IndexType* col_idxs,
//                                  const ValueType* arrow_vals,
//                                  IndexType* row_idxs, IndexType* col_ptrs,
//                                  ValueType* csc_vals,
//                                  UnaryOperator op) GKO_NOT_IMPLEMENTED;

// template <typename ValueType, typename IndexType, typename UnaryOperator>
// void transpose_and_transform(std::shared_ptr<const ReferenceExecutor> exec,
//                              matrix::Arrow<ValueType, IndexType>* trans,
//                              const matrix::Arrow<ValueType, IndexType>* orig,
//                              UnaryOperator op) GKO_NOT_IMPLEMENTED;

// template <typename ValueType, typename IndexType>
// void transpose(std::shared_ptr<const ReferenceExecutor> exec,
//                const matrix::Arrow<ValueType, IndexType>* orig,
//                matrix::Arrow<ValueType, IndexType>* trans)
//                GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


// template <typename ValueType, typename IndexType>
// void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
//                     const matrix::Arrow<ValueType, IndexType>* orig,
//                     matrix::Arrow<ValueType, IndexType>* trans)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


// template <typename ValueType, typename IndexType>
// void calculate_nonzeros_per_row_in_span(
//     std::shared_ptr<const DefaultExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* source, const span& row_span,
//     const span& col_span, array<IndexType>* row_nnz) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);


// template <typename ValueType, typename IndexType>
// void calculate_nonzeros_per_row_in_index_set(
//     std::shared_ptr<const DefaultExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* source,
//     const gko::index_set<IndexType>& row_index_set,
//     const gko::index_set<IndexType>& col_index_set,
//     IndexType* row_nnz) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL);


// template <typename ValueType, typename IndexType>
// void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,
//                        const matrix::Arrow<ValueType, IndexType>* source,
//                        gko::span row_span, gko::span col_span,
//                        matrix::Arrow<ValueType, IndexType>* result)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL);


// template <typename ValueType, typename IndexType>
// void compute_submatrix_from_index_set(
//     std::shared_ptr<const DefaultExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* source,
//     const gko::index_set<IndexType>& row_index_set,
//     const gko::index_set<IndexType>& col_index_set,
//     matrix::Arrow<ValueType, IndexType>* result) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL);


// template <typename ValueType, typename IndexType>
// void convert_to_hybrid(
//     std::shared_ptr<const ReferenceExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* source, const int64*,
//     matrix::Hybrid<ValueType, IndexType>* result) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


// template <typename IndexType>
// void invert_permutation(std::shared_ptr<const DefaultExecutor> exec,
//                         size_type size, const IndexType* permutation_indices,
//                         IndexType* inv_permutation) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


// template <typename ValueType, typename IndexType>
// void inv_symm_permute(
//     std::shared_ptr<const ReferenceExecutor> exec, const IndexType* perm,
//     const matrix::Arrow<ValueType, IndexType>* orig,
//     matrix::Arrow<ValueType, IndexType>* permuted) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


// template <typename ValueType, typename IndexType>
// void row_permute(
//     std::shared_ptr<const ReferenceExecutor> exec, const IndexType* perm,
//     const matrix::Arrow<ValueType, IndexType>* orig,
//     matrix::Arrow<ValueType, IndexType>* row_permuted) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


// template <typename ValueType, typename IndexType>
// void inverse_row_permute(
//     std::shared_ptr<const ReferenceExecutor> exec, const IndexType* perm,
//     const matrix::Arrow<ValueType, IndexType>* orig,
//     matrix::Arrow<ValueType, IndexType>* row_permuted) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


// template <typename ValueType, typename IndexType>
// void inverse_column_permute(
//     std::shared_ptr<const ReferenceExecutor> exec, const IndexType* perm,
//     const matrix::Arrow<ValueType, IndexType>* orig,
//     matrix::Arrow<ValueType, IndexType>* column_permuted)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


// template <typename ValueType, typename IndexType>
// void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
//                           matrix::Arrow<ValueType, IndexType>* to_sort)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


// template <typename ValueType, typename IndexType>
// void is_sorted_by_column_index(
//     std::shared_ptr<const ReferenceExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* to_check,
//     bool* is_sorted) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


// template <typename ValueType, typename IndexType>
// void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
//                       const matrix::Arrow<ValueType, IndexType>* orig,
//                       matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


// template <typename ValueType, typename IndexType>
// void scale(std::shared_ptr<const ReferenceExecutor> exec,
//            const matrix::Dense<ValueType>* alpha,
//            matrix::Arrow<ValueType, IndexType>* to_scale)
//            GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SCALE_KERNEL);


// template <typename ValueType, typename IndexType>
// void inv_scale(std::shared_ptr<const ReferenceExecutor> exec,
//                const matrix::Dense<ValueType>* alpha,
//                matrix::Arrow<ValueType, IndexType>* to_scale)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SCALE_KERNEL);


// template <typename ValueType, typename IndexType>
// void check_diagonal_entries_exist(
//     std::shared_ptr<const ReferenceExecutor> exec,
//     const matrix::Arrow<ValueType, IndexType>* const mtx,
//     bool& has_all_diags) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


// template <typename ValueType, typename IndexType>
// void add_scaled_identity(std::shared_ptr<const ReferenceExecutor> exec,
//                          const matrix::Dense<ValueType>* const alpha,
//                          const matrix::Dense<ValueType>* const beta,
//                          matrix::Arrow<ValueType, IndexType>* const mtx)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL);

// template <typename ValueType, typename IndexType>
// convert_to(
//         Arrow<next_precision<ValueType>, IndexType>* result)
//     GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROW_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void factorize_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks, const IndexType* partitions,
    IndexType* a_cur_row_ptrs,
    const std::vector<std::unique_ptr<LinOp>>* matrices,
    std::vector<std::unique_ptr<LinOp>>* l_factors,
    std::vector<std::unique_ptr<LinOp>>* u_factors,
    ValueType dummy_valuetype_var) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void factorize_off_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, IndexType split_index,
    IndexType num_blocks, const IndexType* partitions,
    std::vector<std::unique_ptr<LinOp>>* triang_factors,
    std::vector<std::unique_ptr<LinOp>>* off_diagonal_blocks,
    ValueType dummy_valuetype_var) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks,
    const IndexType* partitions,
    const std::vector<std::unique_ptr<LinOp>>* l_factors_10,
    const std::vector<std::unique_ptr<LinOp>>* u_factors_01,
    std::vector<std::unique_ptr<LinOp>>* schur_complement_in,
    ValueType dummy_valuetype_var) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL);


}  // namespace arrow
}  // namespace reference
}  // namespace kernels
}  // namespace gko
