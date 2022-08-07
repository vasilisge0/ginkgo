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

#ifndef GKO_CORE_FACTORIZATION_ARROW_LU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_ARROW_LU_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/factorization/arrow_matrix.hpp"

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


const double PIVOT_THRESHOLD = 1e-11;
const double PIVOT_AUGMENTATION = 1e-8;  // officially it is sqrt(eps)*||A||_1


// Helper function kernels.

#define GKO_DECLARE_COMPUTE_REMAINING_NNZ_ROW_CHECK_KERNEL(IndexType) \
    IndexType compute_remaining_nnz_row_check(                        \
        const IndexType* row_ptrs_src, IndexType* row_ptrs_cur,       \
        IndexType row_start, IndexType row_end);

#define GKO_DECLARE_COMPUTE_REMAINING_NNZ_COL_CHECK_KERNEL(IndexType) \
    IndexType compute_remaining_nnz_col_check(                        \
        const IndexType* col_idxs_src, IndexType* row_ptrs_cur,       \
        const IndexType row_start, const IndexType row_end,           \
        const IndexType col_end);

#define GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL(ValueType, IndexType)            \
    void spdgemm_blocks(                                                   \
        std::shared_ptr<const DefaultExecutor>, dim<2> size,               \
        IndexType block_size,                                              \
        const factorization::arrow_submatrix_11<ValueType, IndexType>*     \
            submtx_11,                                                     \
        const factorization::arrow_submatrix_12<ValueType, IndexType>*     \
            submtx_12,                                                     \
        const factorization::arrow_submatrix_21<ValueType, IndexType>*     \
            submtx_21,                                                     \
        matrix::Dense<ValueType>* schur_complement, IndexType block_index, \
        ValueType alpha);

#define GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL(ValueType, IndexType)       \
    void convert_csr_2_dense(                                              \
        dim<2> size, const IndexType* row_ptrs, const IndexType* col_idxs, \
        const ValueType* values, matrix::Dense<ValueType>* dense_mtx,      \
        const IndexType col_start, const IndexType col_end);

#define GKO_DECLARE_FIND_MIN_COL_KERNEL(IndexType)                            \
    void find_min_col(const IndexType* row_ptrs_src,                          \
                      const IndexType* col_idxs_src, IndexType* row_ptrs_cur, \
                      dim<2> size, IndexType row_start, IndexType row_end,    \
                      IndexType* col_min_out, IndexType* row_min_out,         \
                      IndexType* num_occurences_out);

#define GKO_DECLARE_FACTORIZE_KERNEL(ValueType)                \
    void factorize_kernel(const matrix::Dense<ValueType>* mtx, \
                          matrix::Dense<ValueType>* l_factor,  \
                          matrix::Dense<ValueType>* u_factor);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL(ValueType)      \
    void upper_triangular_solve_kernel(dim<2> dim_l_factor,       \
                                       const ValueType* l_factor, \
                                       dim<2> dim_rhs, ValueType* rhs_matrix);

#define GKO_DECLARE_UPPER_TRIANGULAR_LEFT_SOLVE_KERNEL(ValueType)       \
    void upper_triangular_left_solve_kernel(                            \
        dim<2> dim_l_factor, const ValueType* l_factor, dim<2> dim_lhs, \
        ValueType* lhs_matrix);


#define GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL(ValueType)      \
    void lower_triangular_solve_kernel(dim<2> dim_l_factor,       \
                                       const ValueType* l_factor, \
                                       dim<2> dim_rhs, ValueType* rhs_matrix);


// Factorization kernels.

#define GKO_DECLARE_INITIALIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType) \
    void initialize_submatrix_11(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const factorization::arrow_partitions<IndexType>* partitions,    \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_11(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const factorization::arrow_partitions<IndexType>* partitions,   \
        const matrix::Csr<ValueType, IndexType>* mtx,                   \
        factorization::arrow_submatrix_11<ValueType, IndexType>* submtx_11);

#define GKO_DECLARE_PREPROCESS_SUBMATRIX_12_KERNEL(ValueType, IndexType)    \
    void preprocess_submatrix_12(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const factorization::arrow_partitions<IndexType>* partitions,       \
        const matrix::Csr<ValueType, IndexType>* mtx,                       \
        factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12, \
        array<IndexType>& row_ptrs_cur_src_array,                           \
        array<IndexType>& row_ptrs_cur_dst_array);

#define GKO_DECLARE_INITIALIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType)    \
    void initialize_submatrix_12(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const factorization::arrow_partitions<IndexType>* partitions,       \
        const matrix::Csr<ValueType, IndexType>* mtx,                       \
        factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12, \
        array<IndexType>& row_ptrs_cur_src_array);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_12(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const factorization::arrow_partitions<IndexType>* partitions,   \
        const factorization::arrow_submatrix_11<ValueType, IndexType>*  \
            submtx_11,                                                  \
        factorization::arrow_submatrix_12<ValueType, IndexType>* submtx_12);

#define GKO_DECLARE_PREPROCESS_SUBMATRIX_21_KERNEL(ValueType, IndexType)    \
    void preprocess_submatrix_21(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const factorization::arrow_partitions<IndexType>* partitions,       \
        const matrix::Csr<ValueType, IndexType>* mtx,                       \
        factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21, \
        array<IndexType>& col_ptrs_dst_array,                               \
        array<IndexType>& row_ptrs_dst_array);

#define GKO_DECLARE_INITIALIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType) \
    void initialize_submatrix_21(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const factorization::arrow_partitions<IndexType>* partitions,    \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_21(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const factorization::arrow_partitions<IndexType>* partitions,   \
        const factorization::arrow_submatrix_11<ValueType, IndexType>*  \
            submtx_11,                                                  \
        factorization::arrow_submatrix_21<ValueType, IndexType>* submtx_21);

#define GKO_DECLARE_INITIALIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType) \
    void initialize_submatrix_22(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const factorization::arrow_partitions<IndexType>* partitions,    \
        const factorization::arrow_submatrix_11<ValueType, IndexType>*   \
            submtx_11,                                                   \
        const factorization::arrow_submatrix_12<ValueType, IndexType>*   \
            submtx_12,                                                   \
        const factorization::arrow_submatrix_21<ValueType, IndexType>*   \
            submtx_21,                                                   \
        factorization::arrow_submatrix_22<ValueType, IndexType>* submtx_22);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_22(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        factorization::arrow_submatrix_22<ValueType, IndexType>* submtx_22);

#define GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType)   \
    void compute_factors(                                                   \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        factorization::arrow_lu_workspace<ValueType, IndexType>* workspace, \
        const gko::matrix::Csr<ValueType, IndexType>* mtx);


#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename IndexType>                                     \
    GKO_DECLARE_COMPUTE_REMAINING_NNZ_ROW_CHECK_KERNEL(IndexType);    \
    template <typename IndexType>                                     \
    GKO_DECLARE_COMPUTE_REMAINING_NNZ_COL_CHECK_KERNEL(IndexType);    \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL(ValueType, IndexType);          \
    template <typename IndexType>                                     \
    GKO_DECLARE_FIND_MIN_COL_KERNEL(IndexType);                       \
    template <typename ValueType>                                     \
    GKO_DECLARE_UPPER_TRIANGULAR_LEFT_SOLVE_KERNEL(ValueType);        \
    template <typename ValueType>                                     \
    GKO_DECLARE_FACTORIZE_KERNEL(ValueType);                          \
    template <typename ValueType>                                     \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL(ValueType);             \
    template <typename ValueType>                                     \
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL(ValueType);             \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_INITIALIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_PREPROCESS_SUBMATRIX_12_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_INITIALIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_PREPROCESS_SUBMATRIX_21_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_INITIALIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_INITIALIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(arrow_lu, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

// Will have to remove the following from here later.

namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The arrow_lu namespace.
 *
 * @ingroup factor
 */
namespace arrow_lu {

template <typename ValueType, typename IndexType>
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_lu_workspace<ValueType, IndexType>* workspace,
    const gko::matrix::Csr<ValueType, IndexType>* mtx) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

}  // namespace arrow_lu
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The arrow_lu namespace.
 *
 * @ingroup factor
 */
namespace arrow_lu {

template <typename ValueType, typename IndexType>
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_lu_workspace<ValueType, IndexType>* workspace,
    const gko::matrix::Csr<ValueType, IndexType>* mtx) GKO_NOT_IMPLEMENTED;

// GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factors, compute_factors);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

}  // namespace arrow_lu
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ARROW_LU_KERNELS_HPP_
