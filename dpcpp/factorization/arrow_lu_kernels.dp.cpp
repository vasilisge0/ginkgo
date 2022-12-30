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

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>


#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"


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

template <typename ValueType, typename IndexType>
void factorize_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks const IndexType* partitions, IndexType* a_cur_row_ptrs,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> matrices,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> l_factors,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> u_factors)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_LU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);

template <typename ValueType, typename IndexType>
void factorize_off_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, IndexType split_index,
    IndexType num_blocks, const IndexType* partitions,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> a_off_diagonal_blocks,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> triang_factors,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> off_diagonal_blocks)
    GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks,
    const IndexType* partitions,
    const std::vector<std::unique_ptr<LinOp>>* l_factors_10,
    const std::vector<std::unique_ptr<LinOp>>* u_factors_01,
    std::vector<std::unique_ptr<LinOp>>* schur_complement_in,
    ValueType dummy_valuetype_var) GKO_NOT_IMPLEMENTED;

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL);
//

template <typename ValueType, typename IndexType>
void factorize_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks, const IndexType* partitions,
    IndexType* a_cur_row_ptrs,
    const std::shared_ptr<matrix::Dense<ValueType>> matrices,
    std::shared_ptr<matrix::Dense<ValueType>> l_factors,
    std::shared_ptr<matrix::Dense<ValueType>> u_factors)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_SCHUR_COMPLEMENT_KERNEL);

}  // namespace arrow_lu
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
