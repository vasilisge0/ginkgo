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


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/upper_trs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class UpperTrs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;

    UpperTrs() : rand_engine(30), solve_struct_ref{}, solve_struct_omp{} {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::shared_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::shared_ptr<Mtx> gen_u_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_upper_triangular_matrix<Mtx>(
            num_rows, num_cols, false,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        t_b = b->clone();
        t_x = x->clone();
        d_b = gko::clone(omp, b);
        d_x = gko::clone(omp, x);
        dt_b = gko::clone(omp, b);
        dt_x = gko::clone(omp, x);
        mat = gen_u_mtx(m, m);
        csr_mat = CsrMtx::create(ref);
        mat->convert_to(csr_mat.get());
        d_mat = gko::clone(omp, mat);
        d_csr_mat = gko::clone(omp, csr_mat);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::default_random_engine rand_engine;

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx> t_b;
    std::shared_ptr<Mtx> t_x;
    std::shared_ptr<Mtx> mat;
    std::shared_ptr<CsrMtx> csr_mat;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<Mtx> dt_b;
    std::shared_ptr<Mtx> dt_x;
    std::shared_ptr<Mtx> d_mat;
    std::shared_ptr<CsrMtx> d_csr_mat;
    std::shared_ptr<gko::solver::SolveStruct> solve_struct_ref;
    std::shared_ptr<gko::solver::SolveStruct> solve_struct_omp;
};


TEST_F(UpperTrs, OmpUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;
    gko::kernels::omp::upper_trs::should_perform_transpose(omp, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, OmpUpperTrsGenerateIsEquivalentToRef)
{
    gko::size_type num_rhs = 1;
    gko::kernels::reference::upper_trs::generate(ref, csr_mat.get(),
                                                 solve_struct_ref, num_rhs);
    gko::kernels::omp::upper_trs::generate(omp, d_csr_mat.get(),
                                           solve_struct_omp, num_rhs);
}


TEST_F(UpperTrs, OmpUpperTrsSolveIsEquivalentToRef)
{
    initialize_data(59, 43);

    gko::kernels::reference::upper_trs::solve(ref, csr_mat.get(),
                                              solve_struct_ref.get(), t_b.get(),
                                              t_x.get(), b.get(), x.get());
    gko::kernels::omp::upper_trs::solve(omp, d_csr_mat.get(),
                                        solve_struct_omp.get(), dt_b.get(),
                                        dt_x.get(), d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, ApplyIsEquivalentToRef)
{
    initialize_data(59, 3);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(omp);
    auto solver = upper_trs_factory->generate(csr_mat);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mat);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
