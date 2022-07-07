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

#include <ginkgo/core/factorization/arrow_lu.hpp>


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include "core/factorization/arrow_matrix.hpp"


#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ic_kernels.hpp"
#include "core/factorization/par_ict_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"

// #include <ginkgo/core/matrix/arrow.hpp>

namespace gko {
namespace factorization {
namespace arrow_lu {
namespace {

GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);
GKO_REGISTER_OPERATION(init_factor, par_ic_factorization::init_factor);
GKO_REGISTER_OPERATION(compute_factor, par_ic_factorization::compute_factor);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(compute_factors, arrow_lu::compute_factors);
GKO_REGISTER_OPERATION(testr, par_ict_factorization::test);


}  // anonymous namespace
}  // namespace arrow_lu

// template <typename ValueType, typename IndexType>
// std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType,
// IndexType>::generate(
//     const std::shared_ptr<const LinOp>& system_matrix) const
// {
//     std::cout << "testing generate with one argument\n";
// }

template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors) const
{
    // std::cout << "in generate\n";
    // using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    // using CooMatrix = matrix::Coo<ValueType, IndexType>;

    // GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    // const auto exec = this->get_executor();

    // // Converts the system matrix to CSR.
    // // Throws an exception if it is not convertible.
    // auto csr_system_matrix = CsrMatrix::create(exec);
    // as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
    //     ->convert_to(csr_system_matrix.get());
    // // If necessary, sort it
    // if (!skip_sorting) {
    //     csr_system_matrix->sort_by_column_index();
    // }
    // std::cout << "before add a diagonal element\n";
    // // Add explicit diagonal zero elements if they are missing
    // exec->run(arrow_lu::make_add_diagonal_elements(
    //     csr_system_matrix.get(), true));
    // std::cout << "before compute factors\n";
    // std::cout << " * ****** **  \n\n";
    // std::cout <<
    // "parameters_.get_workspace->mtx_.partitions_.get_num_elems(): " <<
    // parameters_.workspace->mtx_.partitions_.data.get_num_elems() << '\n';
    // std::cout <<
    // "parameters_.get_workspace->mtx_.partitions_.data.get_data()[0]: " <<
    // parameters_.workspace->mtx_.partitions_.data.get_data()[0] << '\n';
    // std::cout <<
    // "parameters_.get_workspace->mtx_.partitions_.data.get_data()[0]: " <<
    // parameters_.workspace.get()->mtx_.partitions_.data.get_data()[0] << '\n';
    // // exec->run(arrow_lu::make_compute_factors(csr_system_matrix.get(),
    // parameters_.workspace.get()));

    // std::cout << "after compute factors\n";

    // const auto matrix_size = csr_system_matrix->get_size();
    // const auto number_rows = matrix_size[0];
    // array<IndexType> l_row_ptrs{exec, number_rows + 1};
    // exec->run(arrow_lu::make_initialize_row_ptrs_l(
    //     csr_system_matrix.get(), l_row_ptrs.get_data()));

    // // Get nnz from device memory
    // auto l_nnz = static_cast<size_type>(
    //     exec->copy_val_to_host(l_row_ptrs.get_data() + number_rows));

    // // Since `row_ptrs` of L is already created, the matrix can be
    // // directly created with it
    // array<IndexType> l_col_idxs{exec, l_nnz};
    // array<ValueType> l_vals{exec, l_nnz};
    // std::shared_ptr<CsrMatrix> l_factor = matrix_type::create(
    //     exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
    //     std::move(l_row_ptrs), parameters_.l_strategy);

    // exec->run(par_ic_factorization::make_initialize_l(csr_system_matrix.get(),
    //                                                   l_factor.get(),
    //                                                   false));

    // // build COO representation of lower factor
    // array<IndexType> l_row_idxs{exec, l_nnz};
    // // copy values from l_factor, which are the lower triangular values of A
    // auto l_vals_view = make_array_view(exec, l_nnz, l_factor->get_values());
    // auto a_vals = array<ValueType>{exec, l_vals_view};
    // auto a_row_idxs = array<IndexType>{exec, l_nnz};
    // auto a_col_idxs = make_array_view(exec, l_nnz, l_factor->get_col_idxs());
    // auto a_lower_coo =
    //     CooMatrix::create(exec, matrix_size, std::move(a_vals),
    //                       std::move(a_col_idxs), std::move(a_row_idxs));

    // // compute sqrt of diagonal entries
    // exec->run(par_ic_factorization::make_init_factor(l_factor.get()));

    // // execute sweeps
    // exec->run(par_ic_factorization::make_compute_factor(
    //     parameters_.iterations, a_lower_coo.get(), l_factor.get()));

    // if (both_factors) {
    //     auto lh_factor = l_factor->conj_transpose();
    //     return Composition<ValueType>::create(std::move(l_factor),
    //                                           std::move(lh_factor));
    // } else {
    //     return Composition<ValueType>::create(std::move(l_factor));
    // }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors, arrow_partitions<IndexType>& partitions,
    std::shared_ptr<arrow_lu_workspace<ValueType, IndexType>> workspace) const
{
    // std::cout << "in generate\n";
    // using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    // using CooMatrix = matrix::Coo<ValueType, IndexType>;

    // GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    // const auto exec = this->get_executor();

    // // Converts the system matrix to CSR.
    // // Throws an exception if it is not convertible.
    // auto csr_system_matrix = share(CsrMatrix::create(exec));
    // as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
    //     ->convert_to(csr_system_matrix.get());
    // // If necessary, sort it
    // if (!skip_sorting) {
    //     csr_system_matrix->sort_by_column_index();
    // }

    // // Add explicit diagonal zero elements if they are missing
    // // exec->run(arrow_lu::make_add_diagonal_elements(
    // //     csr_system_matrix.get(), true));

    // std::cout << "<< testing >>" << "\n";
    // exec->run(arrow_lu::make_compute_factors(csr_system_matrix.get(),
    // workspace.get())); exec->run(arrow_lu::make_testr(1));

    // const auto matrix_size = csr_system_matrix->get_size();
    // const auto number_rows = matrix_size[0];
    // array<IndexType> l_row_ptrs{exec, number_rows + 1};
    // exec->run(arrow_lu::make_initialize_row_ptrs_l(
    //     csr_system_matrix.get(), l_row_ptrs.get_data()));

    // // Get nnz from device memory
    // auto l_nnz = static_cast<size_type>(
    //     exec->copy_val_to_host(l_row_ptrs.get_data() + number_rows));

    // // Since `row_ptrs` of L is already created, the matrix can be
    // // directly created with it
    // array<IndexType> l_col_idxs{exec, l_nnz};
    // array<ValueType> l_vals{exec, l_nnz};
    // std::shared_ptr<CsrMatrix> l_factor = matrix_type::create(
    //     exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
    //     std::move(l_row_ptrs), parameters_.l_strategy);

    // exec->run(par_ic_factorization::make_initialize_l(csr_system_matrix.get(),
    //                                                   l_factor.get(),
    //                                                   false));

    // // build COO representation of lower factor
    // array<IndexType> l_row_idxs{exec, l_nnz};
    // // copy values from l_factor, which are the lower triangular values of A
    // auto l_vals_view = make_array_view(exec, l_nnz, l_factor->get_values());
    // auto a_vals = array<ValueType>{exec, l_vals_view};
    // auto a_row_idxs = array<IndexType>{exec, l_nnz};
    // auto a_col_idxs = make_array_view(exec, l_nnz, l_factor->get_col_idxs());
    // auto a_lower_coo =
    //     CooMatrix::create(exec, matrix_size, std::move(a_vals),
    //                       std::move(a_col_idxs), std::move(a_row_idxs));

    // // compute sqrt of diagonal entries
    // exec->run(par_ic_factorization::make_init_factor(l_factor.get()));

    // // execute sweeps
    // exec->run(par_ic_factorization::make_compute_factor(
    //     parameters_.iterations, a_lower_coo.get(), l_factor.get()));

    // if (both_factors) {
    //     auto lh_factor = l_factor->conj_transpose();
    //     return Composition<ValueType>::create(std::move(l_factor),
    //                                           std::move(lh_factor));
    // } else {
    //     return Composition<ValueType>::create(std::move(l_factor));
    // }
}

template <typename ValueType, typename IndexType>
void ArrowLu<ValueType, IndexType>::generate_workspace(
    const std::shared_ptr<const LinOp>& system_matrix)
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto csr_system_matrix = CsrMatrix::create(exec);
    as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        ->convert_to(csr_system_matrix.get());
    exec->run(arrow_lu::make_compute_factors(csr_system_matrix.get(),
                                             parameters_.workspace.get()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors, std::ifstream& infile) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto csr_system_matrix = CsrMatrix::create(exec);
    as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        ->convert_to(csr_system_matrix.get());
    // If necessary, sort it
    if (!skip_sorting) {
        csr_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(
        arrow_lu::make_add_diagonal_elements(csr_system_matrix.get(), true));
    // const auto matrix_size = csr_system_matrix->get_size();
    // const auto number_rows = matrix_size[0];
    // array<IndexType> l_row_ptrs{exec, number_rows + 1};
    // exec->run(arrow_lu::make_initialize_row_ptrs_l(
    //     csr_system_matrix.get(), l_row_ptrs.get_data()));

    // // Get nnz from device memory
    // auto l_nnz = static_cast<size_type>(
    //     exec->copy_val_to_host(l_row_ptrs.get_data() + number_rows));

    // // Since `row_ptrs` of L is already created, the matrix can be
    // // directly created with it
    // array<IndexType> l_col_idxs{exec, l_nnz};
    // array<ValueType> l_vals{exec, l_nnz};
    // std::shared_ptr<CsrMatrix> l_factor = matrix_type::create(
    //     exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
    //     std::move(l_row_ptrs), parameters_.l_strategy);

    // exec->run(par_ic_factorization::make_initialize_l(csr_system_matrix.get(),
    //                                                   l_factor.get(),
    //                                                   false));

    // // build COO representation of lower factor
    // array<IndexType> l_row_idxs{exec, l_nnz};
    // // copy values from l_factor, which are the lower triangular values of A
    // auto l_vals_view = make_array_view(exec, l_nnz, l_factor->get_values());
    // auto a_vals = array<ValueType>{exec, l_vals_view};
    // auto a_row_idxs = array<IndexType>{exec, l_nnz};
    // auto a_col_idxs = make_array_view(exec, l_nnz, l_factor->get_col_idxs());
    // auto a_lower_coo =
    //     CooMatrix::create(exec, matrix_size, std::move(a_vals),
    //                       std::move(a_col_idxs), std::move(a_row_idxs));

    // // compute sqrt of diagonal entries
    // exec->run(par_ic_factorization::make_init_factor(l_factor.get()));

    // // execute sweeps
    // exec->run(par_ic_factorization::make_compute_factor(
    //     parameters_.iterations, a_lower_coo.get(), l_factor.get()));

    // if (both_factors) {
    //     auto lh_factor = l_factor->conj_transpose();
    //     return Composition<ValueType>::create(std::move(l_factor),
    //                                           std::move(lh_factor));
    // } else {
    //     return Composition<ValueType>::create(std::move(l_factor));
    // }
}

#define GKO_DECLARE_ARROWLU(ValueType, IndexType) \
    class ArrowLu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROWLU);


}  // namespace factorization
}  // namespace gko
