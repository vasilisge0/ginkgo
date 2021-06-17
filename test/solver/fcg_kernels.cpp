/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/solver/fcg.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/fcg_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Fcg : public ::testing::Test {
protected:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Fcg<value_type>;

    Fcg() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx.get());
        return result;
    }

    void initialize_data()
    {
        gko::size_type m = 597;
        gko::size_type n = 43;
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        t = gen_mtx(m, n, n + 2);
        z = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        q = gen_mtx(m, n, n + 2);
        x = gen_mtx(m, n, n + 3);
        beta = gen_mtx(1, n, n);
        prev_rho = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
        rho_t = gen_mtx(1, n, n);
        // check correct handling for zero values
        beta->at(2) = 0.0;
        prev_rho->at(2) = 0.0;
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = Mtx::create(exec);
        d_b->copy_from(b.get());
        d_r = Mtx::create(exec);
        d_r->copy_from(r.get());
        d_t = Mtx::create(exec);
        d_t->copy_from(t.get());
        d_z = Mtx::create(exec);
        d_z->copy_from(z.get());
        d_p = Mtx::create(exec);
        d_p->copy_from(p.get());
        d_q = Mtx::create(exec);
        d_q->copy_from(q.get());
        d_x = Mtx::create(exec);
        d_x->copy_from(x.get());
        d_beta = Mtx::create(exec);
        d_beta->copy_from(beta.get());
        d_prev_rho = Mtx::create(exec);
        d_prev_rho->copy_from(prev_rho.get());
        d_rho_t = Mtx::create(exec);
        d_rho_t->copy_from(rho_t.get());
        d_rho = Mtx::create(exec);
        d_rho->copy_from(rho.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(exec, n));
        *d_stop_status = *stop_status;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> t;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> rho_t;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_t;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_rho_t;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Fcg, FcgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::initialize(
        ref, b.get(), r.get(), z.get(), p.get(), q.get(), t.get(),
        prev_rho.get(), rho.get(), rho_t.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::initialize(
        exec, d_b.get(), d_r.get(), d_z.get(), d_p.get(), d_q.get(), d_t.get(),
        d_prev_rho.get(), d_rho.get(), d_rho_t.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho_t, rho_t, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Fcg, FcgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::step_1(ref, p.get(), z.get(), rho_t.get(),
                                         prev_rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::step_1(exec, d_p.get(), d_z.get(),
                                              d_rho_t.get(), d_prev_rho.get(),
                                              d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
}


TEST_F(Fcg, FcgStep2IsEquivalentToRef)
{
    initialize_data();
    gko::kernels::reference::fcg::step_2(ref, x.get(), r.get(), t.get(),
                                         p.get(), q.get(), beta.get(),
                                         rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::step_2(
        exec, d_x.get(), d_r.get(), d_t.get(), d_p.get(), d_q.get(),
        d_beta.get(), d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
}


TEST_F(Fcg, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 53);
    gko::test::make_hpd(mtx.get());
    auto x = gen_mtx(50, 3, 4);
    auto b = gen_mtx(50, 3, 5);
    auto d_mtx = Mtx::create(exec);
    d_mtx->copy_from(mtx.get());
    auto d_x = Mtx::create(exec);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(exec);
    d_b->copy_from(b.get());
    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(ref),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(ref))
            .on(ref);
    auto d_fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(exec))
            .on(exec);
    auto solver = fcg_factory->generate(std::move(mtx));
    auto d_solver = d_fcg_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 100);
}


}  // namespace
