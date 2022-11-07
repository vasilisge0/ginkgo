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

#include "core/factorization/arrow_lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
// #include "core/factorization/arrow_matrix.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The ArrowLu namespace.
 *
 * @ingroup factor
 */
namespace arrow_lu {


constexpr int default_block_size = 512;

// template <typename ValueType, typename IndexType>
// void compute_factors(
//    std::shared_ptr<const DefaultExecutor> exec,
//    matrix::Csr<ValueType, IndexType>* global_mtx,
//    gko::factorization::arrow_lu_workspace<ValueType, IndexType>* workspace)
//    GKO_NOT_IMPLEMENTED;
//
template <typename ValueType, typename IndexType>
void compute_factors(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_lu_workspace<ValueType, IndexType>* workspace,
    const gko::matrix::Csr<ValueType, IndexType>* mtx) GKO_NOT_IMPLEMENTED;

// GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factors, compute_factors);

template <typename ValueType, typename IndexType>
void factorize_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, dim<2> size,
    IndexType num_blocks, const IndexType* partitions,
    IndexType* a_cur_row_ptrs,
    const std::vector<std::unique_ptr<LinOp>>* matrices,
    std::vector<std::unique_ptr<LinOp>>* l_factors,
    std::vector<std::unique_ptr<LinOp>>* u_factors,
    ValueType dummy_valuetype_var)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_DIAGONAL_SUBMATRIX_KERNEL);

// Step 3 of computing LU factors of submatrix_12. Sets up the
// nonzero entries of submatrix_12 of U factor.
template <typename ValueType, typename IndexType>
void factorize_off_diagonal_submatrix(
    std::shared_ptr<const DefaultExecutor> exec, IndexType split_index,
    IndexType num_blocks, const IndexType* partitions,
    std::vector<std::unique_ptr<LinOp>>* a_off_diagonal_blocks,
    std::vector<std::unique_ptr<LinOp>>* triang_factors,
    std::vector<std::unique_ptr<LinOp>>* off_diagonal_blocks,
    ValueType dummy_valuetype_var)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_FACTORIZE_OFF_DIAGONAL_SUBMATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_schur_complement(
    std::shared_ptr<const DefaultExecutor> exec, IndexType num_blocks,
    const IndexType* partitions,
    const std::vector<std::unique_ptr<LinOp>>* l_factors_10,
    const std::vector<std::unique_ptr<LinOp>>* u_factors_01,
    std::vector<std::unique_ptr<LinOp>>* schur_complement_in,
    ValueType dummy_valuetype_var)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROWLU_COMPUTE_SCHUR_COMPLEMENT_KERNEL);


// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_ARROW_LU_COMPUTE_FACTORS_KERNEL);

}  // namespace arrow_lu
}  // namespace hip
}  // namespace kernels
}  // namespace gko
