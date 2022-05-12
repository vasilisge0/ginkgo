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

#ifndef GKO_PUBLIC_CORE_LOG_SOLVER_PROGRESS_HPP_
#define GKO_PUBLIC_CORE_LOG_SOLVER_PROGRESS_HPP_


#include <fstream>
#include <iostream>


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


template <typename ValueType = default_precision>
class SolverProgress : public Logger {
public:
    /* Internal solver events */
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution = nullptr,
        const LinOp* residual_norm = nullptr) const override;

    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override;

    static std::unique_ptr<SolverProgress> create(
        std::shared_ptr<const Executor> exec, std::ostream& os = std::cout)
    {
        return std::unique_ptr<SolverProgress>(new SolverProgress(exec, os));
    }

protected:
    /**
     * Creates a Stream logger.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     */
    explicit SolverProgress(std::shared_ptr<const gko::Executor> exec,
                            std::ostream& os)
        : Logger(exec, Logger::iteration_complete_mask), os_(os)
    {}


private:
    std::ostream& os_;
    static constexpr const char* prefix_ = "[SOLVER] >>> ";
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_SOLVER_PROGRESS_HPP_
