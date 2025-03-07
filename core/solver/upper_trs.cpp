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

#include <ginkgo/core/solver/upper_trs.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>


#include "core/solver/upper_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace upper_trs {
namespace {


GKO_REGISTER_OPERATION(generate, upper_trs::generate);
GKO_REGISTER_OPERATION(should_perform_transpose,
                       upper_trs::should_perform_transpose);
GKO_REGISTER_OPERATION(solve, upper_trs::solve);


}  // anonymous namespace
}  // namespace upper_trs


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(const UpperTrs& other)
    : EnableLinOp<UpperTrs>(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(UpperTrs&& other)
    : EnableLinOp<UpperTrs>(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    const UpperTrs& other)
{
    if (this != &other) {
        EnableLinOp<UpperTrs>::operator=(other);
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(other);
        this->generate();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    UpperTrs&& other)
{
    if (this != &other) {
        EnableLinOp<UpperTrs>::operator=(std::move(other));
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(std::move(other));
        if (this->get_executor() == other.get_executor()) {
            this->solve_struct_ = std::exchange(other.solve_struct_, nullptr);
        } else {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->transpose()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::conj_transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->conj_transpose()));
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::generate()
{
    if (this->get_system_matrix()) {
        this->get_executor()->run(
            upper_trs::make_generate(this->get_system_matrix().get(),
                                     this->solve_struct_, parameters_.num_rhs));
    }
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            const auto exec = this->get_executor();

            // This kernel checks if a transpose is needed for the multiple rhs
            // case. Currently only the algorithm for CUDA version <=9.1 needs
            // this transposition due to the limitation in the cusparse
            // algorithm. The other executors (omp and reference) do not use the
            // transpose (trans_x and trans_b) and hence are passed in empty
            // pointers.
            bool do_transpose = false;
            std::shared_ptr<Vector> trans_b;
            std::shared_ptr<Vector> trans_x;
            this->get_executor()->run(
                upper_trs::make_should_perform_transpose(do_transpose));
            if (do_transpose) {
                trans_b =
                    Vector::create(exec, gko::transpose(dense_b->get_size()));
                trans_x =
                    Vector::create(exec, gko::transpose(dense_x->get_size()));
            } else {
                trans_b = Vector::create(exec);
                trans_x = Vector::create(exec);
            }
            exec->run(upper_trs::make_solve(
                lend(this->get_system_matrix()), lend(this->solve_struct_),
                lend(trans_b), lend(trans_x), dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                const LinOp* b,
                                                const LinOp* beta,
                                                LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_UPPER_TRS(_vtype, _itype) class UpperTrs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS);


}  // namespace solver
}  // namespace gko
