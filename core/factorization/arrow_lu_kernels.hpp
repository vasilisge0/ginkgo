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

// template<typename ValueType, typename IndexType> struct arrow_matrix;
// template<typename ValueType, typename IndexType> struct arrow_submatrix_11;
// template<typename ValueType, typename IndexType> struct arrow_submatrix_12;
// template<typename ValueType, typename IndexType> struct arrow_submatrix_21;
// template<typename ValueType, typename IndexType> struct arrow_submatrix_22;
// template<typename IndexType> struct arrow_partitions;

#define GKO_DECLARE_SYMBOLIC_COUNT_ROW_CHECK_KERNEL(IndexType)         \
    IndexType symbolic_count_row_check(                                \
        IndexType* row_ptrs_current, const IndexType* row_ptrs_source, \
        const IndexType row_start, const IndexType row_end);

#define GKO_DECLARE_SYMBOLIC_COUNT_COL_CHECK_KERNEL(IndexType)         \
    IndexType symbolic_count_col_check(                                \
        IndexType* row_ptrs_current, const IndexType* col_idxs_source, \
        const IndexType row_start, const IndexType row_end,            \
        const IndexType col_end);

#define GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL(ValueType, IndexType)       \
    void spdgemm_blocks(                                              \
        const dim<2> size, const IndexType block_size,                \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>& \
            submtx_11,                                                \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>& \
            submtx_12,                                                \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>& \
            submtx_21,                                                \
        gko::factorization::arrow_submatrix_22<ValueType, IndexType>& \
            submtx_22,                                                \
        const IndexType block_index, const ValueType alpha);

#define GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL(ValueType, IndexType)       \
    void convert_csr_2_dense(                                              \
        dim<2> size, const IndexType* row_ptrs, const IndexType* col_idxs, \
        const ValueType* values, matrix::Dense<ValueType>* dense_mtx,      \
        const IndexType col_start, const IndexType col_end);

#define GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL(ValueType)                \
    void lower_triangular_solve_kernel(dim<2> dim_l_factor,                 \
                                       ValueType* l_factor, dim<2> dim_rhs, \
                                       ValueType* rhs_matrix);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL_KERNEL(ValueType)         \
    void upper_triangular_solve_kernel(dim<2> dim_l_factor,                 \
                                       ValueType* l_factor, dim<2> dim_rhs, \
                                       ValueType* rhs_matrix);

#define GKO_DECLARE_CSC_SPDGEMM_KERNEL(ValueType, IndexType)            \
    void csc_spdgemm(const dim<2> dim_mtx, const ValueType* values_mtx, \
                     const dim<2> dim_rhs, ValueType* values_rhs,       \
                     const ValueType alpha);

#define GKO_DECLARE_CSR_SPDGEMM_KERNEL(ValueType, IndexType)            \
    void csr_spdgemm(const dim<2> dim_mtx, const ValueType* values_mtx, \
                     const dim<2> dim_rhs, ValueType* values_rhs,       \
                     const ValueType alpha);

#define GKO_DECLARE_FIND_MIN_COL_12_KERNEL(ValueType, IndexType)        \
    void find_min_col(                                                  \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&   \
            submtx_12,                                                  \
        const IndexType* row_ptrs, const IndexType* col_idxs,           \
        IndexType row_start, IndexType row_end, IndexType* col_min_out, \
        IndexType* row_min_out, IndexType* num_occurences_out);

#define GKO_DECLARE_FIND_MIN_COL_21_KERNEL(ValueType, IndexType)        \
    void find_min_col(                                                  \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&   \
            submtx_21,                                                  \
        const IndexType* row_ptrs, const IndexType* col_idxs,           \
        IndexType row_start, IndexType row_end, IndexType* col_min_out, \
        IndexType* row_min_out, IndexType* num_occurences_out);

#define GKO_DECLARE_STEP_1_IMPL_ASSEMBLE_11_KERNEL(ValueType, IndexType) \
    void step_1_impl_assemble(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,             \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&    \
            submtx_11,                                                   \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_12_KERNEL(ValueType, IndexType) \
    void step_1_impl_symbolic_count(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,                   \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&          \
            submtx_12,                                                         \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_21_KERNEL(ValueType, IndexType) \
    void step_1_impl_symbolic_count(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,                   \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&          \
            submtx_21,                                                         \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_1_IMPL_COMPUTE_SCHUR_COMPLEMENT_KERNEL(ValueType, \
                                                                IndexType) \
    void step_1_impl_compute_schur_complement(                             \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&      \
            submtx_11,                                                     \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&      \
            submtx_12,                                                     \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&      \
            submtx_21,                                                     \
        gko::factorization::arrow_submatrix_22<ValueType, IndexType>&      \
            submtx_22,                                                     \
        gko::factorization::arrow_partitions<IndexType>& arrow_partitions);

#define GKO_DECLARE_STEP_2_IMPL_FACTORIZE_11_KERNEL(ValueType, IndexType) \
    void step_2_impl_factorize(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,              \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&     \
            submtx_11,                                                    \
        gko::factorization::arrow_partitions<IndexType>& partitions)

#define GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_12_KERNEL(ValueType, IndexType) \
    void step_2_impl_assemble(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,             \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&    \
            submtx_12,                                                   \
        gko::factorization::arrow_partitions<IndexType>& arrow_partitions);

#define GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_21_KERNEL(ValueType, IndexType) \
    void step_2_impl_assemble(                                           \
        const matrix::Csr<ValueType, IndexType>* global_mtx,             \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&    \
            submtx_21,                                                   \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_3_IMPL_FACTORIZE_12_KERNEL(ValueType, IndexType) \
    void step_3_impl_factorize(                                           \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&     \
            submtx_11,                                                    \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&     \
            submtx_12,                                                    \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_3_IMPL_FACTORIZE_21_KERNEL(ValueType, IndexType) \
    void step_3_impl_factorize(                                           \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&     \
            submtx_11,                                                    \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&     \
            submtx_21,                                                    \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_STEP_2_IMPL_FACTORIZE_22_KERNEL(ValueType, IndexType) \
    void step_2_impl_factorize(                                           \
        gko::factorization::arrow_submatrix_22<ValueType, IndexType>&     \
            submtx_22);

#define GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_1_KERNEL(ValueType, IndexType) \
    void lower_triangular_solve_step_1(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_2_KERNEL(ValueType, IndexType) \
    void lower_triangular_solve_step_2(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_3_KERNEL(ValueType, IndexType) \
    void lower_triangular_solve_step_3(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_1_KERNEL(ValueType, IndexType) \
    void upper_triangular_solve_step_1(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_2_KERNEL(ValueType, IndexType) \
    void upper_triangular_solve_step_2(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_3_KERNEL(ValueType, IndexType) \
    void upper_triangular_solve_step_3(                                        \
        gko::factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,  \
        matrix::Dense<ValueType>* rhs);

#define GKO_DECLARE_FACTORIZE_KERNEL(ValueType)               \
    void factorize_kernel(matrix::Dense<ValueType>* mtx,      \
                          matrix::Dense<ValueType>* l_factor, \
                          matrix::Dense<ValueType>* u_factor);

#define GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL(ValueType)                \
    void upper_triangular_solve_kernel(dim<2> dim_l_factor,                 \
                                       ValueType* l_factor, dim<2> dim_rhs, \
                                       ValueType* rhs_matrix);

#define GKO_DECLARE_UPPER_TRIANGULAR_LEFT_SOLVE_KERNEL(ValueType) \
    void upper_triangular_left_solve_kernel(                      \
        dim<2> dim_l_factor, ValueType* l_factor, dim<2> dim_lhs, \
        ValueType* lhs_matrix);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_11(                                        \
        const matrix::Csr<ValueType, IndexType>* global_mtx,            \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&   \
            submtx_11,                                                  \
        gko::factorization::arrow_partitions<IndexType>& partitions);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_12(                                        \
        const matrix::Csr<ValueType, IndexType>* global_mtx,            \
        gko::factorization::arrow_partitions<IndexType>& partitions,    \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&   \
            submtx_11,                                                  \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&   \
            submtx_12);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_21(                                        \
        const matrix::Csr<ValueType, IndexType>* global_mtx,            \
        gko::factorization::arrow_partitions<IndexType>& partitions,    \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&   \
            submtx_11,                                                  \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&   \
            submtx_21);

#define GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType) \
    void factorize_submatrix_22(                                        \
        const matrix::Csr<ValueType, IndexType>* global_mtx,            \
        gko::factorization::arrow_partitions<IndexType>& partitions,    \
        gko::factorization::arrow_submatrix_11<ValueType, IndexType>&   \
            submtx_11,                                                  \
        gko::factorization::arrow_submatrix_12<ValueType, IndexType>&   \
            submtx_12,                                                  \
        gko::factorization::arrow_submatrix_21<ValueType, IndexType>&   \
            submtx_21,                                                  \
        gko::factorization::arrow_submatrix_22<ValueType, IndexType>&   \
            submtx_22);

#define GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType) \
    void compute_factors(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        gko::matrix::Csr<ValueType, IndexType>* global_mtx,               \
        gko::factorization::arrow_lu_workspace<ValueType, IndexType>*     \
            workspace);


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename IndexType>                                           \
    GKO_DECLARE_SYMBOLIC_COUNT_ROW_CHECK_KERNEL(IndexType);                 \
    template <typename IndexType>                                           \
    GKO_DECLARE_SYMBOLIC_COUNT_COL_CHECK_KERNEL(IndexType);                 \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPDGEMM_BLOCKS_KERNEL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_CONVERT_CSR_2_DENSE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType>                                           \
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_KERNEL(ValueType);                   \
    template <typename ValueType>                                           \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL_KERNEL(ValueType);            \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_CSC_SPDGEMM_KERNEL(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_CSR_SPDGEMM_KERNEL(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FIND_MIN_COL_12_KERNEL(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FIND_MIN_COL_21_KERNEL(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_1_IMPL_ASSEMBLE_11_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_12_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_1_IMPL_SYMBOLIC_COUNT_21_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_1_IMPL_COMPUTE_SCHUR_COMPLEMENT_KERNEL(ValueType,      \
                                                            IndexType);     \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_11_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_12_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_2_IMPL_ASSEMBLE_21_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_3_IMPL_FACTORIZE_12_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_3_IMPL_FACTORIZE_21_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_STEP_2_IMPL_FACTORIZE_22_KERNEL(ValueType, IndexType);      \
    template <typename ValueType>                                           \
    GKO_DECLARE_UPPER_TRIANGULAR_LEFT_SOLVE_KERNEL(ValueType);              \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_1_KERNEL(ValueType, size_type); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_2_KERNEL(ValueType, size_type); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_3_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_1_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_2_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_3_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                           \
    GKO_DECLARE_FACTORIZE_KERNEL(ValueType);                                \
    template <typename ValueType>                                           \
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_KERNEL(ValueType);                   \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_11_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_12_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_21_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_FACTORIZE_SUBMATRIX_22_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType)
                                         // template <typename IndexType> \
    // GKO_DECLARE_TEST_KERNEL(IndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(arrow_lu, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

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
    gko::matrix::Csr<ValueType, IndexType>* global_mtx,
    gko::factorization::arrow_lu_workspace<ValueType, IndexType>* workspace)
    GKO_NOT_IMPLEMENTED;

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
    matrix::Csr<ValueType, IndexType>* global_mtx,
    gko::factorization::arrow_lu_workspace<ValueType, IndexType>* workspace)
    GKO_NOT_IMPLEMENTED;

// GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factors, compute_factors);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

}  // namespace arrow_lu
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ARROW_LU_KERNELS_HPP_
