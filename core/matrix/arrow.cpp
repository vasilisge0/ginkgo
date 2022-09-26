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

#include <ginkgo/core/matrix/arrow.hpp>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {

// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
// override GKO_NOT_IMPLEMENTED;

// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp*
// b, const LinOp* beta,
//                 LinOp* x) const override GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
Arrow<ValueType, IndexType>& Arrow<ValueType, IndexType>::operator=(
    const Arrow<ValueType, IndexType>& other) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
Arrow<ValueType, IndexType>& Arrow<ValueType, IndexType>::operator=(
    Arrow<ValueType, IndexType>&& other) GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
const IndexType* Arrow<ValueType, IndexType>::get_const_partition_idxs() const
{
    return partitions_.get_const_data();
}

template <typename ValueType, typename IndexType>
size_type Arrow<ValueType, IndexType>::get_partitions_num_elems() const
{
    return partitions_.get_num_elems();
}

template <typename ValueType, typename IndexType>
void Arrow<ValueType, IndexType>::set_partitions(
    array<IndexType>& partitions_in)
{
    this->partitions_ = std::move(partitions_in);
}

template <typename ValueType, typename IndexType>
IndexType Arrow<ValueType, IndexType>::get_num_blocks()
{
    return partitions_.get_num_elems() - 1;
}

template <typename ValueType, typename IndexType>
std::shared_ptr<std::vector<std::unique_ptr<gko::LinOp>>>
Arrow<ValueType, IndexType>::get_submatrix_00()
{
    return submtx_00_;
}

template <typename ValueType, typename IndexType>
std::shared_ptr<std::vector<std::unique_ptr<gko::LinOp>>>
Arrow<ValueType, IndexType>::get_submatrix_01()
{
    return submtx_01_;
}

template <typename ValueType, typename IndexType>
std::shared_ptr<std::vector<std::unique_ptr<gko::LinOp>>>
Arrow<ValueType, IndexType>::get_submatrix_10()
{
    return submtx_10_;
}

template <typename ValueType, typename IndexType>
std::shared_ptr<std::vector<std::unique_ptr<gko::LinOp>>>
Arrow<ValueType, IndexType>::get_submatrix_11()
{
    return submtx_11_;
}

// template <typename ValueType, typename IndexType>
// Arrow<ValueType, IndexType>::Arrow(const Arrow<ValueType, IndexType>& other)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// Arrow<ValueType, IndexType>::Arrow(Arrow<ValueType, IndexType>&& other)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp*
// b,
//                                             const LinOp* beta, LinOp* x)
//                                             const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void matrix::Arrow<ValueType, IndexType>::convert_to(
//    matrix::Arrow<next_precision<ValueType>, IndexType>* result) const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(
//    Arrow<next_precision<ValueType>, IndexType>* result) GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    Coo<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Coo<ValueType, IndexType>* result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Dense<ValueType>* result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    Hybrid<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Hybrid<ValueType, IndexType>*
// result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    Sellp<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Sellp<ValueType, IndexType>*
// result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    SparsityCsr<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(
//    SparsityCsr<ValueType, IndexType>* result) GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    Ell<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Ell<ValueType, IndexType>* result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    Fbcsr<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Fbcsr<ValueType, IndexType>*
// result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::convert_to(
//    matrix::Arrow<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::move_to(Arrow<ValueType, IndexType>*
// result)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::read(const mat_data& data)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::read(const device_mat_data& data)
//    GKO_NOT_IMPLEMENTED;
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::read(device_mat_data&& data)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::write(mat_data& data) const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::transpose() const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::conj_transpose() const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::inverse_permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::row_permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::column_permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::inverse_row_permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<LinOp> Arrow<ValueType, IndexType>::inverse_column_permute(
//    const array<IndexType>* permutation_indices) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::sort_by_column_index() GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// bool Arrow<ValueType, IndexType>::is_sorted_by_column_index() const
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<Arrow<ValueType, IndexType>>
// Arrow<ValueType, IndexType>::create_submatrix(
//    const gko::span& row_span,
//    const gko::span& column_span) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<Arrow<ValueType, IndexType>>
// Arrow<ValueType, IndexType>::create_submatrix(
//    const index_set<IndexType>& row_index_set,
//    const index_set<IndexType>& col_index_set) const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<Diagonal<ValueType>>
// Arrow<ValueType, IndexType>::extract_diagonal() const GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::compute_absolute_inplace()
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// std::unique_ptr<typename Arrow<ValueType, IndexType>::absolute_type>
// Arrow<ValueType, IndexType>::compute_absolute() const GKO_NOT_IMPLEMENTED;
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::scale_impl(const LinOp* alpha)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::inv_scale_impl(const LinOp* alpha)
//    GKO_NOT_IMPLEMENTED;
//
//
// template <typename ValueType, typename IndexType>
// void Arrow<ValueType, IndexType>::add_scaled_identity_impl(
//    const LinOp* const a, const LinOp* const b) GKO_NOT_IMPLEMENTED;

#define GKO_DECLARE_ARROW_MATRIX(ValueType, IndexType) \
    class Arrow<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROW_MATRIX);


}  // namespace matrix
}  // namespace gko
