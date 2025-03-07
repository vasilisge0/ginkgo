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

#include "core/factorization/par_ic_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel IC factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::Csr<ValueType, IndexType>* l)
{
    auto num_rows = l->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_vals = l->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto l_nz = l_row_ptrs[row + 1] - 1;
        auto diag = sqrt(l_vals[l_nz]);
        if (is_finite(diag)) {
            l_vals[l_nz] = diag;
        } else {
            l_vals[l_nz] = one<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    const matrix::Coo<ValueType, IndexType>* a_lower,
                    matrix::Csr<ValueType, IndexType>* l)
{
    auto num_rows = a_lower->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_values();
    auto a_vals = a_lower->get_const_values();

    for (size_type i = 0; i < iterations; ++i) {
#pragma omp parallel for
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type l_nz = l_row_ptrs[row]; l_nz < l_row_ptrs[row + 1];
                 ++l_nz) {
                auto col = l_col_idxs[l_nz];
                auto a_val = a_vals[l_nz];
                // accumulate l(row,:) * l(col,:) without the last entry l(col,
                // col)
                ValueType sum{};
                auto l_begin = l_row_ptrs[row];
                auto l_end = l_row_ptrs[row + 1];
                auto lh_begin = l_row_ptrs[col];
                auto lh_end = l_row_ptrs[col + 1];
                while (l_begin < l_end && lh_begin < lh_end) {
                    auto l_col = l_col_idxs[l_begin];
                    auto lh_row = l_col_idxs[lh_begin];
                    if (l_col == lh_row && l_col < col) {
                        sum += l_vals[l_begin] * conj(l_vals[lh_begin]);
                    }
                    l_begin += (l_col <= lh_row);
                    lh_begin += (lh_row <= l_col);
                }
                auto new_val = a_val - sum;
                if (row == col) {
                    new_val = sqrt(new_val);
                } else {
                    auto diag = l_vals[l_row_ptrs[col + 1] - 1];
                    new_val = new_val / diag;
                }
                if (is_finite(new_val)) {
                    l_vals[l_nz] = new_val;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
