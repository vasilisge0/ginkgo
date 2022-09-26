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

// #include <ginkgo/core/factorization/arrow_lu.hpp>


// #include <memory>

// #include <ginkgo/core/base/array.hpp>
// #include <ginkgo/core/base/composition.hpp>
// #include <ginkgo/core/base/lin_op.hpp>
// #include <ginkgo/core/base/exception_helpers.hpp>
// #include <ginkgo/core/base/polymorphic_object.hpp>
// #include <ginkgo/core/base/types.hpp>
// #include <ginkgo/core/matrix/coo.hpp>
// #include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>


// #include "core/components/format_conversion_kernels.hpp"
// #include "core/factorization/arrow_lu_kernels.hpp"
// #include "core/factorization/factorization_kernels.hpp"
// #include "core/factorization/par_ic_kernels.hpp"
// #include "core/factorization/par_ict_kernels.hpp"
// #include "core/matrix/csr_kernels.hpp"


// #include <ginkgo/core/factorization/ilu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>


// #include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"


// #include <ginkgo/core/matrix/arrow.hpp>

namespace gko {
namespace factorization {
namespace arrow_lu {
namespace {

GKO_REGISTER_OPERATION(factorize_diagonal_submatrix,
                       arrow_lu::factorize_diagonal_submatrix);
GKO_REGISTER_OPERATION(factorize_off_diagonal_submatrix,
                       arrow_lu::factorize_off_diagonal_submatrix);
GKO_REGISTER_OPERATION(compute_schur_complement,
                       arrow_lu::compute_schur_complement);

}  // anonymous namespace
}  // namespace arrow_lu


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix,
    array<IndexType>& partitions) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;
    using ArrowMatrix = matrix::Arrow<ValueType, IndexType>;

    // GKO_ASSERT_IS_SQUARE_MATRIX(arrow_system_matrix);

    // Converts the system matrix to Arrow.
    // Throws an exception if it is not convertible.
    const auto exec = this->get_executor();

    // conversion should be performed beforehand
    // auto arrow_system_matrix = ArrowMatrix::create(exec, partitions);
    auto arrow_system_matrix =
        as<ArrowMatrix>(system_matrix);  // (!) note: dynamic_cast
    const auto partition_idxs = arrow_system_matrix->get_const_partition_idxs();
    const auto num_blocks = arrow_system_matrix->get_partitions_num_elems() - 1;
    const auto split_index = partition_idxs[num_blocks];
    array<IndexType> a_cur_row_ptrs = {exec, num_blocks + 1};
    std::shared_ptr<matrix::Arrow<ValueType, IndexType>> l_factor;
    std::shared_ptr<matrix::Arrow<ValueType, IndexType>> u_factor;

    // Wraps-up submatrices of arrow_system_matrix and l_factor and u_factor in
    // collection_of_matrices objects.
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        blocks_of_submatrix_00(arrow_system_matrix->get_submatrix_00().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        l_factors_of_submatrix_00(l_factor->get_submatrix_00().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        u_factors_of_submatrix_00(u_factor->get_submatrix_00().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        u_factors_of_submatrix_01(u_factor->get_submatrix_01().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        l_factors_of_submatrix_10(l_factor->get_submatrix_10().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        blocks_of_submatrix_11(arrow_system_matrix->get_submatrix_11().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        u_factors_of_submatrix_11(u_factor->get_submatrix_11().get());
    gko::factorization::arrow_lu::collection_of_matrices<ValueType>
        l_factors_of_submatrix_11(l_factor->get_submatrix_11().get());

    // Factorizes blocks of submatrix_00.
    exec->run(arrow_lu::make_factorize_diagonal_submatrix(
        arrow_system_matrix->get_size(), static_cast<IndexType>(num_blocks),
        partition_idxs, a_cur_row_ptrs.get_data(), &blocks_of_submatrix_00,
        &l_factors_of_submatrix_00, &u_factors_of_submatrix_00));

    // Factorizes blocks of submatrix_01.
    exec->run(arrow_lu::make_factorize_off_diagonal_submatrix(
        split_index, static_cast<IndexType>(num_blocks), partition_idxs,
        &l_factors_of_submatrix_00, &u_factors_of_submatrix_01));

    // Factorizes blocks of submatrix_10.
    exec->run(arrow_lu::make_factorize_off_diagonal_submatrix(
        split_index, static_cast<IndexType>(num_blocks), partition_idxs,
        &u_factors_of_submatrix_00, &l_factors_of_submatrix_10));

    // Computes schur complement.
    exec->run(arrow_lu::make_compute_schur_complement(
        static_cast<IndexType>(num_blocks), partition_idxs,
        &l_factors_of_submatrix_10, &u_factors_of_submatrix_01,
        &blocks_of_submatrix_11));

    // Factorizes submatrix_11
    IndexType one = 1;
    array<IndexType> partitions_last = {exec, 2};
    partitions_last.get_data()[0] = partitions.get_data()[num_blocks];
    partitions_last.get_data()[0] = arrow_system_matrix->get_size()[0];
    exec->run(arrow_lu::make_factorize_diagonal_submatrix(
        arrow_system_matrix->get_size(), static_cast<IndexType>(1),
        partitions_last.get_data(), a_cur_row_ptrs.get_data(),
        &blocks_of_submatrix_11, &l_factors_of_submatrix_11,
        &u_factors_of_submatrix_11));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}

#define GKO_DECLARE_ARROWLU_KERNEL(ValueType, IndexType) \
    class ArrowLu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROWLU_KERNEL);


}  // namespace factorization
}  // namespace gko
