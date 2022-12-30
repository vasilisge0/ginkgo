/*******************************<GINK GO LICENSE>******************************
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


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>


#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"


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
GKO_REGISTER_OPERATION(factorize_schur_complement,
                       arrow_lu::factorize_schur_complement);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);

}  // anonymous namespace
}  // namespace arrow_lu


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ArrowLu<ValueType, IndexType>::compute_factors(
    const std::shared_ptr<const LinOp>& system_matrix) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;
    using ArrowMatrix = matrix::Arrow<ValueType, IndexType>;

    const auto exec = this->get_executor();
    auto arrow_system_matrix = as<ArrowMatrix>(system_matrix);
    const auto partition_idxs = arrow_system_matrix->get_const_partition_idxs();
    auto num_blocks = arrow_system_matrix->get_partitions_num_elems() - 1;
    const auto split_index = partition_idxs[num_blocks];
    array<IndexType> a_cur_row_ptrs = {exec, num_blocks + 1};

    // Initializes l_fator, u_factor.
    array<IndexType> p1 = {exec, num_blocks + 1};
    exec->copy(num_blocks + 1, arrow_system_matrix->get_const_partition_idxs(),
               p1.get_data());
    array<IndexType> p2 = {exec, num_blocks + 1};
    exec->copy(num_blocks + 1, arrow_system_matrix->get_const_partition_idxs(),
               p2.get_data());
    auto l_factor =
        share(gko::matrix::Arrow<value_type, index_type>::create(exec, p1));
    auto u_factor =
        share(gko::matrix::Arrow<value_type, index_type>::create(exec, p2));

    // Computes submatrix_00 of l_factor and u_factor.
    exec->run(arrow_lu::make_factorize_diagonal_submatrix(
        arrow_system_matrix->get_size(), static_cast<IndexType>(num_blocks),
        partition_idxs, a_cur_row_ptrs.get_data(),
        arrow_system_matrix->get_submatrix_00(), l_factor->get_submatrix_00(),
        u_factor->get_submatrix_00()));

    // Computes submatrix_01 of u_factor.
    exec->run(arrow_lu::make_factorize_off_diagonal_submatrix(
        split_index, static_cast<IndexType>(num_blocks), partition_idxs,
        arrow_system_matrix->get_submatrix_01(), l_factor->get_submatrix_00(),
        u_factor->get_submatrix_01()));

    // Computes submatrix_10 of l_factor.
    exec->run(arrow_lu::make_factorize_off_diagonal_submatrix(
        split_index, static_cast<IndexType>(num_blocks), partition_idxs,
        arrow_system_matrix->get_submatrix_10(), u_factor->get_submatrix_00(),
        l_factor->get_submatrix_10()));

    // Computes schur complement.
    exec->run(arrow_lu::make_compute_schur_complement(
        static_cast<IndexType>(num_blocks), partition_idxs,
        l_factor->get_submatrix_10(), u_factor->get_submatrix_01(),
        arrow_system_matrix->get_submatrix_11()));

    // Computes submatrix_11 of l_factor and u_factor.
    IndexType one = 1;
    array<IndexType> partitions_last = {exec, 2};
    partitions_last.get_data()[0] = partition_idxs[num_blocks];
    partitions_last.get_data()[1] = arrow_system_matrix->get_size()[0];
    auto len = arrow_system_matrix->get_size()[0] - partition_idxs[num_blocks];
    gko::dim<2> schur_size = {len, len};
    exec->run(arrow_lu::make_factorize_schur_complement(
        schur_size, static_cast<IndexType>(num_blocks), partition_idxs,
        a_cur_row_ptrs.get_data(),
        as<matrix::Dense<ValueType>>(arrow_system_matrix->get_submatrix_11()),
        as<matrix::Dense<ValueType>>(l_factor->get_submatrix_11()),
        as<matrix::Dense<ValueType>>(u_factor->get_submatrix_11())));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}

#define GKO_DECLARE_ARROWLU_KERNEL(ValueType, IndexType) \
    class ArrowLu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROWLU_KERNEL);


}  // namespace factorization
}  // namespace gko
