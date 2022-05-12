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

#include <ginkgo/core/log/solver_progress.hpp>


#include <iomanip>


#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace log {


template <typename ValueType>
void SolverProgress<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm) const
{
    this->on_iteration_complete(solver, num_iterations, residual, solution,
                                residual_norm, nullptr);
}


template <typename ValueType>
void SolverProgress<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    os_ << prefix_ << "iteration" << std::setw(4) << num_iterations;
    if (auto base = dynamic_cast<const solver::SolverBase<LinOp>*>(solver)) {
        auto scalars = base->get_workspace_scalars();
        auto names = base->get_workspace_op_names();
        for (auto scalar : scalars) {
            if (auto dense = dynamic_cast<const matrix::Dense<ValueType>*>(
                    base->get_workspace_op(scalar))) {
                os_ << std::setw(10) << names[scalar] << std::setw(15)
                    << std::setprecision(8)
                    << solver->get_executor()->copy_val_to_host(
                           dense->get_const_values());
            }
        }
    }
    os_ << '\n';
}


#define GKO_DECLARE_SOLVER_PROGRESS(_type) class SolverProgress<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SOLVER_PROGRESS);


}  // namespace log
}  // namespace gko
