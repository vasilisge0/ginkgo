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
// #include <ginkgo/core/factorization/arrow_lu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/factorization/arrow_matrix.hpp"

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {

const double PIVOT_THRESHOLD = 1e-11;
const double PIVOT_AUGMENTATION = 1e-8;  // officially it is sqrt(eps)*||A||_1

// // Factorization kernels.


// #define GKO_DECLARE_GENERATE_ARROWLU_KERNEL(ValueType, IndexType) \
// std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate( \
//     const std::shared_ptr<const Lin>& arrow_system_matrix) const;

// #define GKO_DECLARE_ALL_AS_TEMPLATES                  \
//     template <typename ValueType, typename IndexType> \
//     GKO_DECLARE_GENERATE_ARROWLU_KERNEL(ValueType, IndexType)

// GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(arrow_lu,
//     GKO_DECLARE_ALL_AS_TEMPLATES);

// #undef GKO_DECLARE_ALL_AS_TEMPLATES


// #define GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType) \
//     void compute_factors(                                                 \
//         std::shared_ptr<const DefaultExecutor> exec,                      \
//         factorization::ArrowLuState<ValueType, IndexType>* workspace,     \
//         const gko::matrix::Csr<ValueType, IndexType>* mtx);


// #define GKO_DECLARE_ALL_AS_TEMPLATES                  \
//     template <typename ValueType, typename IndexType> \
//     GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL(ValueType, IndexType)

// GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(arrow_lu,
// GKO_DECLARE_ALL_AS_TEMPLATES);


// #undef GKO_DECLARE_ALL_AS_TEMPLATES


// }  // namespace kernels
// }  // namespace gko


#define GKO_DECLARE_ARROWLU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL(ValueType,     \
                                                                IndexType)     \
    void factorize_diagonal_submatrix(                                         \
        std::shared_ptr<const DefaultExecutor> exec, dim<2> size,              \
        IndexType num_blocks, const IndexType* partitions,                     \
        IndexType* a_cur_row_ptrs,                                             \
        const factorization::arrow_lu::collection_of_matrices<ValueType>*      \
            matrices,                                                          \
        factorization::arrow_lu::collection_of_matrices<ValueType>* l_factors, \
        factorization::arrow_lu::collection_of_matrices<ValueType>*            \
            u_factors);

#define GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL(ValueType, \
                                                                    IndexType) \
    void factorize_off_diagonal_submatrix(                                     \
        std::shared_ptr<const DefaultExecutor> exec, IndexType split_index,    \
        IndexType num_blocks, const IndexType* partitions,                     \
        const factorization::arrow_lu::collection_of_matrices<ValueType>*      \
            matrices,                                                          \
        factorization::arrow_lu::collection_of_matrices<ValueType>*            \
            triang_factors);

#define GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL(ValueType,     \
                                                            IndexType)     \
    void compute_schur_complement(                                         \
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks, \
        const IndexType* partitions,                                       \
        const factorization::arrow_lu::collection_of_matrices<ValueType>*  \
            l_factors_10,                                                  \
        const factorization::arrow_lu::collection_of_matrices<ValueType>*  \
            u_factors_01,                                                  \
        factorization::arrow_lu::collection_of_matrices<ValueType>*        \
            schur_complement_in);

#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_ARROWLU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL(ValueType,     \
                                                            IndexType)     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL(ValueType, \
                                                                IndexType) \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL(ValueType, IndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(arrow_lu, GKO_DECLARE_ALL_AS_TEMPLATES);

#undef GKO_DECLARE_ALL_AS_TEMPLATES

// namespace dpcpp {
// /**
//  * @brief The arrow_lu namespace.
//  *
//  * @ingroup factor
//  */
// namespace arrow_lu {

// template <typename ValueType, typename IndexType>
// void factorize_diagonal_submatrix(
//     std::shared_ptr<const DefaultExecutor> exec,
//     dim<2> size,
//     IndexType num_blocks,
//     const IndexType* partitions,
//     IndexType* a_cur_row_ptrs,
//     const LinOp* a_linop,
//     LinOp* l_factors,
//     LinOp* u_factors) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROW_LU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);


// }  // namespace arrow_lu
// }  // namespace dpcpp

// namespace hip {
// /**
//  * @brief The arrow_lu namespace.
//  *
//  * @ingroup factor
//  */
// namespace arrow_lu {

// template <typename ValueType, typename IndexType>
// void factorize_diagonal_submatrix(
//     std::shared_ptr<const DefaultExecutor> exec,
//     dim<2> size,
//     IndexType num_blocks,
//     const IndexType* partitions,
//     IndexType* a_cur_row_ptrs,
//     const LinOp* a_linop,
//     LinOp* l_factors,
//     LinOp* u_factors) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROW_LU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);

// }  // namespace arrow_lu
// }  // namespace hip


// GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factors, compute_factors);

}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ARROW_LU_KERNELS_HPP_
