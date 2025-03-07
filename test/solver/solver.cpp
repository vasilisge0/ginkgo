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

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename SolverType>
struct SimpleSolverTest {
    using solver_type = SolverType;
    using value_type = typename solver_type::value_type;
    using index_type = gko::int32;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using precond_type = gko::preconditioner::Jacobi<value_type, index_type>;

    static bool is_iterative() { return true; }

    static bool is_preconditionable() { return true; }

    static double tolerance() { return 1e4 * r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the matrix is well-conditioned
        gko::utils::make_hpd(data, 2.0);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build().with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(iteration_count)
                .on(exec));
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                precond_type::build().with_max_block_size(1u).on(exec));
    }

    static const gko::LinOp* get_preconditioner(const solver_type* solver)
    {
        return solver->get_preconditioner().get();
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        const solver_type* solver)
    {
        return solver->get_stop_criterion_factory().get();
    }

    static void assert_empty_state(const solver_type* mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_system_matrix(), nullptr);
        ASSERT_EQ(mtx->get_preconditioner(), nullptr);
        ASSERT_EQ(mtx->get_stopping_criterion_factory(), nullptr);
    }

    static constexpr bool logs_iteration_complete() { return true; }
};


struct Cg : SimpleSolverTest<gko::solver::Cg<solver_value_type>> {};


struct Cgs : SimpleSolverTest<gko::solver::Cgs<solver_value_type>> {
    static double tolerance() { return 1e5 * r<value_type>::value; }
};


struct Fcg : SimpleSolverTest<gko::solver::Fcg<solver_value_type>> {
    static double tolerance() { return 1e7 * r<value_type>::value; }
};


struct Bicg : SimpleSolverTest<gko::solver::Bicg<solver_value_type>> {};


struct Bicgstab : SimpleSolverTest<gko::solver::Bicgstab<solver_value_type>> {
    // I give up ._. Some cases still have huge differences
    static double tolerance() { return 1e12 * r<value_type>::value; }
};


template <unsigned dimension>
struct Idr : SimpleSolverTest<gko::solver::Idr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_deterministic(true)
            .with_subspace_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_deterministic(true)
            .with_preconditioner(
                precond_type::build().with_max_block_size(1u).on(exec))
            .with_subspace_dim(dimension);
    }
};


struct Ir : SimpleSolverTest<gko::solver::Ir<solver_value_type>> {
    static double tolerance() { return 1e5 * r<value_type>::value; }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_solver(
                precond_type::build().with_max_block_size(1u).on(exec));
    }

    static const gko::LinOp* get_preconditioner(const solver_type* solver)
    {
        return solver->get_solver().get();
    }
};


template <unsigned dimension>
struct CbGmres : SimpleSolverTest<gko::solver::CbGmres<solver_value_type>> {
    static double tolerance() { return 1e9 * r<value_type>::value; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                precond_type::build().with_max_block_size(1u).on(exec))
            .with_krylov_dim(dimension);
    }
};


template <unsigned dimension>
struct Gmres : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                precond_type::build().with_max_block_size(1u).on(exec))
            .with_krylov_dim(dimension);
    }
};


struct LowerTrs : SimpleSolverTest<gko::solver::LowerTrs<solver_value_type>> {
    static bool is_iterative() { return false; }

    static bool is_preconditionable() { return false; }

    static double tolerance() { return r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the diagonal is nonzero
        gko::utils::make_hpd(data, 1.2);
        gko::utils::make_lower_triangular(data);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build();
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type)
    {
        assert(false);
        return solver_type::build();
    }

    static const gko::LinOp* get_preconditioner(const solver_type* solver)
    {
        return nullptr;
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        const solver_type* solver)
    {
        return nullptr;
    }

    static constexpr bool logs_iteration_complete() { return false; }
};


struct UpperTrs : SimpleSolverTest<gko::solver::UpperTrs<solver_value_type>> {
    static bool is_iterative() { return false; }

    static bool is_preconditionable() { return false; }

    static double tolerance() { return r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the diagonal is nonzero
        gko::utils::make_hpd(data, 1.2);
        gko::utils::make_upper_triangular(data);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build();
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type)
    {
        assert(false);
        return solver_type::build();
    }

    static const gko::LinOp* get_preconditioner(const solver_type* solver)
    {
        return nullptr;
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        const solver_type* solver)
    {
        return nullptr;
    }

    static constexpr bool logs_iteration_complete() { return false; }
};


template <typename ObjectType>
struct test_pair {
    std::shared_ptr<ObjectType> ref;
    std::shared_ptr<ObjectType> dev;

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::shared_ptr<const gko::Executor> exec)
        : ref{std::move(ref_obj)}, dev{gko::clone(exec, ref)}
    {}

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::unique_ptr<ObjectType> dev_obj)
        : ref{std::move(ref_obj)}, dev{std::move(dev_obj)}
    {}

    test_pair() = default;
    test_pair(const test_pair& o) = default;
    test_pair(test_pair&& o) noexcept = default;
    test_pair& operator=(const test_pair& o) = default;
    test_pair& operator=(test_pair&& o) noexcept = default;
};


struct DummyLogger : gko::log::Logger {
    DummyLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(std::move(exec),
                           gko::log::Logger::iteration_complete_mask)
    {}

    void on_iteration_complete(const gko::LinOp* solver,
                               const gko::size_type& it, const gko::LinOp* r,
                               const gko::LinOp* x = nullptr,
                               const gko::LinOp* tau = nullptr) const override
    {
        iteration_complete++;
    }

    mutable int iteration_complete = 0;
};


template <typename T>
class Solver : public ::testing::Test {
protected:
    using Config = T;
    using SolverType = typename T::solver_type;
    using Precond = typename T::precond_type;
    using Mtx = typename T::matrix_type;
    using index_type = gko::int32;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;

    Solver() { reset_rand(); }

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        logger = std::make_shared<DummyLogger>(exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    void reset_rand() { rand_engine.seed(15); }

    test_pair<Mtx> gen_mtx(int num_rows, int num_cols, int min_cols,
                           int max_cols)
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                num_rows, num_cols,
                std::uniform_int_distribution<>(min_cols, max_cols),
                std::normal_distribution<>(0.0, 1.0), rand_engine);
        Config::preprocess(data);
        auto mtx = Mtx::create(ref);
        mtx->read(data);
        return test_pair<Mtx>{std::move(mtx), exec};
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            rand_engine};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_in_vec(const test_pair<SolverType>& mtx, int nrhs,
                                  int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[1],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_scalar()
    {
        return {gko::initialize<VecType>(
                    {gko::test::detail::get_rand_value<
                        typename VecType::value_type>(
                        std::normal_distribution<
                            gko::remove_complex<typename VecType::value_type>>(
                            0.0, 1.0),
                        rand_engine)},
                    ref),
                exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_out_vec(const test_pair<SolverType>& mtx, int nrhs,
                                   int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[0],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType>
    double tol(const test_pair<VecType>& x)
    {
        return Config::tolerance() *
               std::sqrt(x.ref->get_size()[1] *
                         (gko::is_complex<typename VecType::value_type>()
                              ? 2.0
                              : 1.0));
    }

    template <typename VecType>
    double mixed_tol(const test_pair<VecType>& x)
    {
        return std::max(
            r_mixed<value_type, mixed_value_type>() *
                std::sqrt(x.ref->get_size()[1] *
                          (gko::is_complex<typename VecType::value_type>()
                               ? 2.0
                               : 1.0)),
            tol(x));
    }

    template <typename TestFunction>
    void forall_matrix_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Empty matrix (0x0)");
            guarded_fn(gen_mtx(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Sparse Matrix with variable row nnz (50x50)");
            guarded_fn(gen_mtx(50, 50, 10, 20));
        }
    }

    template <typename TestFunction>
    void forall_solver_scenarios(const test_pair<Mtx>& mtx, TestFunction fn)
    {
        auto guarded_fn = [&](auto solver) {
            try {
                fn(std::move(solver));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Defaulted solver");
            guarded_fn(test_pair<SolverType>{Config::build(ref, 0)
                                                 .on(ref)
                                                 ->generate(mtx.ref)
                                                 ->create_default(),
                                             Config::build(exec, 0)
                                                 .on(exec)
                                                 ->generate(mtx.dev)
                                                 ->create_default()});
        }
        {
            SCOPED_TRACE("Cleared solver");
            test_pair<SolverType> pair{
                Config::build(ref, 0).on(ref)->generate(mtx.ref),
                Config::build(exec, 0).on(exec)->generate(mtx.dev)};
            pair.ref->clear();
            pair.dev->clear();
            guarded_fn(std::move(pair));
        }
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations via clone");
            guarded_fn(test_pair<SolverType>{
                Config::build(ref, 0).on(ref)->generate(mtx.ref), exec});
        }
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build(ref, 0).on(ref)->generate(mtx.ref),
                Config::build(exec, 0).on(exec)->generate(mtx.dev)});
        }
        if (Config::is_preconditionable()) {
            SCOPED_TRACE("Preconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build_preconditioned(ref, 0).on(ref)->generate(mtx.ref),
                Config::build_preconditioned(exec, 0).on(exec)->generate(
                    mtx.dev)});
        }
        if (Config::is_iterative()) {
            {
                SCOPED_TRACE("Unpreconditioned solver with 4 iterations");
                guarded_fn(test_pair<SolverType>{
                    Config::build(ref, 4).on(ref)->generate(mtx.ref),
                    Config::build(exec, 4).on(exec)->generate(mtx.dev)});
            }
            if (Config::is_preconditionable()) {
                SCOPED_TRACE("Preconditioned solver with 4 iterations");
                guarded_fn(test_pair<SolverType>{
                    Config::build_preconditioned(ref, 4).on(ref)->generate(
                        mtx.ref),
                    Config::build_preconditioned(exec, 4).on(exec)->generate(
                        mtx.dev)});
            }
        }
    }

    template <typename VecType = Vec, typename TestFunction>
    void forall_vector_scenarios(const test_pair<SolverType>& solver,
                                 TestFunction fn)
    {
        auto guarded_fn = [&](auto b, auto x) {
            try {
                fn(std::move(b), std::move(x));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Multivector with 0 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 0, 0),
                       gen_out_vec<VecType>(solver, 0, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<VecType>(solver, 1, 1),
                       gen_out_vec<VecType>(solver, 1, 1));
        }
        /*if (Config::is_iterative() && solver.ref->get_system_matrix()) {
            SCOPED_TRACE("Single vector with correct initial guess");
            auto in = gen_in_vec<VecType>(solver, 1, 1);
            auto out = gen_out_vec<VecType>(solver, 1, 1);
            solver.ref->get_system_matrix()->apply(out.ref.get(), in.ref.get());
            solver.dev->get_system_matrix()->apply(out.dev.get(), in.dev.get());
            guarded_fn(std::move(in), std::move(out));
        }*/
        {
            SCOPED_TRACE("Single strided vector");
            guarded_fn(gen_in_vec<VecType>(solver, 1, 2),
                       gen_out_vec<VecType>(solver, 1, 3));
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<VecType>;
            {
                SCOPED_TRACE("Single strided complex vector");
                guarded_fn(gen_in_vec<complex_vec>(solver, 1, 2),
                           gen_out_vec<complex_vec>(solver, 1, 3));
            }
            {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                guarded_fn(gen_in_vec<complex_vec>(solver, 2, 3),
                           gen_out_vec<complex_vec>(solver, 2, 4));
            }
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 2, 2),
                       gen_out_vec<VecType>(solver, 2, 2));
        }
        {
            SCOPED_TRACE("Strided multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 2, 3),
                       gen_out_vec<VecType>(solver, 2, 4));
        }
        {
            SCOPED_TRACE("Multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 40, 40),
                       gen_out_vec<VecType>(solver, 40, 40));
        }
        {
            SCOPED_TRACE("Strided multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 40, 43),
                       gen_out_vec<VecType>(solver, 40, 45));
        }
    }

    void assert_empty_state(const SolverType* solver,
                            std::shared_ptr<const gko::Executor> expected_exec)
    {
        ASSERT_FALSE(solver->get_size());
        ASSERT_EQ(solver->get_executor(), expected_exec);
        ASSERT_EQ(solver->get_system_matrix(), nullptr);
        ASSERT_EQ(Config::get_preconditioner(solver), nullptr);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::shared_ptr<DummyLogger> logger;

    std::default_random_engine rand_engine;
};

using SolverTypes =
    ::testing::Types<Cg, Cgs, Fcg, Bicg, Bicgstab,
                     /* "IDR uses different initialization approaches even when
                        deterministic", Idr<1>, Idr<4>,*/
                     Ir, CbGmres<2>, CbGmres<10>, Gmres<2>, Gmres<10>, LowerTrs,
                     UpperTrs>;

TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, ApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            this->forall_vector_scenarios(
                solver, [this, &solver](auto b, auto x) {
                    solver.ref->apply(b.ref.get(), x.ref.get());
                    solver.dev->apply(b.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol(x));
                });
        });
    });
}


TYPED_TEST(Solver, AdvancedApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            this->forall_vector_scenarios(
                solver, [this, &solver](auto b, auto x) {
                    auto alpha = this->gen_scalar();
                    auto beta = this->gen_scalar();

                    solver.ref->apply(alpha.ref.get(), b.ref.get(),
                                      beta.ref.get(), x.ref.get());
                    solver.dev->apply(alpha.dev.get(), b.dev.get(),
                                      beta.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol(x));
                });
        });
    });
}


TYPED_TEST(Solver, MixedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            this->template forall_vector_scenarios<MixedVec>(
                solver, [this, &solver](auto b, auto x) {
                    solver.ref->apply(b.ref.get(), x.ref.get());
                    solver.dev->apply(b.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol(x));
                });
        });
    });
}


TYPED_TEST(Solver, MixedAdvancedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            this->template forall_vector_scenarios<MixedVec>(
                solver, [this, &solver](auto b, auto x) {
                    auto alpha = this->template gen_scalar<MixedVec>();
                    auto beta = this->template gen_scalar<MixedVec>();

                    solver.ref->apply(alpha.ref.get(), b.ref.get(),
                                      beta.ref.get(), x.ref.get());
                    solver.dev->apply(alpha.dev.get(), b.dev.get(),
                                      beta.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol(x));
                });
        });
    });
}


TYPED_TEST(Solver, CrossExecutorGenerateCopiesToFactoryExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto mtx) {
        auto solver =
            Config::build(this->ref, 0).on(this->exec)->generate(mtx.ref);

        ASSERT_EQ(solver->get_system_matrix()->get_executor(), this->exec);
        ASSERT_EQ(solver->get_executor(), this->exec);
        if (Config::is_iterative()) {
            ASSERT_EQ(Config::get_stop_criterion_factory(solver.get())
                          ->get_executor(),
                      this->exec);
        }
        if (Config::is_preconditionable()) {
            auto precond = Config::get_preconditioner(solver.get());
            ASSERT_EQ(precond->get_executor(), this->exec);
            ASSERT_TRUE(dynamic_cast<
                        const gko::matrix::Identity<typename Mtx::value_type>*>(
                precond));
        }
        GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(solver->get_system_matrix()), mtx.ref,
                            0.0);
    });
}


TYPED_TEST(Solver, CopyAssignSameExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));

            auto& result = (*solver2 = *solver.dev);

            ASSERT_EQ(&result, solver2.get());
            ASSERT_EQ(solver2->get_size(), solver.dev->get_size());
            ASSERT_EQ(solver2->get_executor(), solver.dev->get_executor());
            ASSERT_EQ(solver2->get_system_matrix(),
                      solver.dev->get_system_matrix());
            ASSERT_EQ(Config::get_stop_criterion_factory(solver2.get()),
                      Config::get_stop_criterion_factory(solver.dev.get()));
            ASSERT_EQ(Config::get_preconditioner(solver2.get()),
                      Config::get_preconditioner(solver.dev.get()));
        });
    });
}


TYPED_TEST(Solver, MoveAssignSameExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto in_mtx) {
        this->forall_solver_scenarios(in_mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));
            auto size = solver.dev->get_size();
            auto mtx = solver.dev->get_system_matrix();
            auto precond = Config::get_preconditioner(solver.dev.get());
            auto stop = Config::get_stop_criterion_factory(solver.dev.get());

            auto& result = (*solver2 = std::move(*solver.dev));

            ASSERT_EQ(&result, solver2.get());
            // moved-to object
            ASSERT_EQ(solver2->get_size(), size);
            ASSERT_EQ(solver2->get_executor(), this->exec);
            ASSERT_EQ(solver2->get_system_matrix(), mtx);
            ASSERT_EQ(Config::get_stop_criterion_factory(solver2.get()), stop);
            ASSERT_EQ(Config::get_preconditioner(solver2.get()), precond);
            // moved-from object
            this->assert_empty_state(solver.dev.get(), this->exec);
        });
    });
}


TYPED_TEST(Solver, CopyAssignCrossExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Precond = typename TestFixture::Precond;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));

            auto& result = (*solver2 = *solver.ref);

            ASSERT_EQ(&result, solver2.get());
            ASSERT_EQ(solver2->get_size(), solver.ref->get_size());
            ASSERT_EQ(solver2->get_executor(), this->exec);
            if (solver.ref->get_system_matrix()) {
                GKO_ASSERT_MTX_NEAR(
                    gko::as<Mtx>(solver2->get_system_matrix()),
                    gko::as<Mtx>(solver.ref->get_system_matrix()), 0.0);
                // TODO no easy way to compare stopping criteria cross-executor
                auto precond = Config::get_preconditioner(solver2.get());
                if (dynamic_cast<const Precond*>(precond)) {
                    GKO_ASSERT_MTX_NEAR(
                        gko::as<Precond>(precond),
                        gko::as<Precond>(
                            Config::get_preconditioner(solver.ref.get())),
                        0.0);
                }
            }
        });
    });
}


TYPED_TEST(Solver, MoveAssignCrossExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Precond = typename TestFixture::Precond;
    this->forall_matrix_scenarios([this](auto in_mtx) {
        this->forall_solver_scenarios(in_mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));
            auto size = solver.ref->get_size();
            auto mtx = solver.ref->get_system_matrix();
            auto precond = Config::get_preconditioner(solver.ref.get());
            auto stop = Config::get_stop_criterion_factory(solver.ref.get());

            auto& result = (*solver2 = std::move(*solver.ref));

            ASSERT_EQ(&result, solver2.get());
            // moved-to object
            ASSERT_EQ(solver2->get_size(), size);
            ASSERT_EQ(solver2->get_executor(), this->exec);
            if (solver.ref->get_system_matrix()) {
                GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(solver2->get_system_matrix()),
                                    gko::as<Mtx>(mtx), 0.0);
                // TODO no easy way to compare stopping criteria cross-executor
                auto new_precond = Config::get_preconditioner(solver2.get());
                if (dynamic_cast<const Precond*>(new_precond)) {
                    GKO_ASSERT_MTX_NEAR(gko::as<Precond>(new_precond),
                                        gko::as<Precond>(precond), 0.0);
                }
            }
            // moved-from object
            this->assert_empty_state(solver.ref.get(), this->ref);
        });
    });
}


TYPED_TEST(Solver, ClearIsEmpty)
{
    using Config = typename TestFixture::Config;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            solver.dev->clear();

            this->assert_empty_state(solver.dev.get(), this->exec);
        });
    });
}


TYPED_TEST(Solver, CreateDefaultIsEmpty)
{
    using Config = typename TestFixture::Config;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto default_solver = solver.dev->create_default();

            this->assert_empty_state(default_solver.get(), this->exec);
        });
    });
}


TYPED_TEST(Solver, LogsIterationComplete)
{
    using Config = typename TestFixture::Config;
    if (Config::logs_iteration_complete()) {
        using Mtx = typename TestFixture::Mtx;
        using Vec = typename TestFixture::Vec;
        auto mtx = gko::share(Mtx::create(this->exec));
        auto b = Vec::create(this->exec);
        auto x = Vec::create(this->exec);
        gko::size_type num_iteration(4);
        auto solver = Config::build(this->exec, num_iteration)
                          .on(this->exec)
                          ->generate(mtx);
        auto before_logger = *this->logger;
        solver->add_logger(this->logger);

        solver->apply(b.get(), x.get());

        ASSERT_EQ(this->logger->iteration_complete,
                  before_logger.iteration_complete + num_iteration + 1);
    }
}
