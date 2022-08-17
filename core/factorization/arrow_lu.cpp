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


}  // anonymous namespace
}  // namespace arrow_lu

// void

template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ArrowLu<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix,
    array<IndexType>& partitions)
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto arrow_system_matrix = ArrowMatrix::create(exec);
    as<ConvertibleTo<ArrowMatrix>>(system_matrix.get())
        ->convert_to(arrow_system_matrix.get());

    // exec->run(arrow_lu::make_compute_factors(parameters_.workspace.get(),
    //                                          csr_system_matrix.get()));
}

// template <typename ValueType,
#define GKO_DECLARE_ARROWLU(ValueType, IndexType) \
    class ArrowLu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROWLU);


}  // namespace factorization
}  // namespace gko
