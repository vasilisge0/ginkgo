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


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class UpperTrs : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using mtx_type = gko::matrix::Csr<value_type, index_type>;
    using vec_type = gko::matrix::Dense<>;
    using solver_type = gko::solver::UpperTrs<value_type, index_type>;

    UpperTrs() : rand_engine(30) {}

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

    std::unique_ptr<vec_type> gen_vec(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<vec_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<mtx_type> gen_u_mtx(int size, int row_nnz)
    {
        return gko::test::generate_random_upper_triangular_matrix<mtx_type>(
            size, false, std::uniform_int_distribution<>(row_nnz, size),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<mtx_type> gen_mtx(int size, int row_nnz)
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                size, size, std::uniform_int_distribution<>(row_nnz, size),
                std::normal_distribution<>(-1.0, 1.0), rand_engine);
        gko::utils::make_diag_dominant(data);
        auto result = mtx_type::create(ref);
        result->read(data);
        return result;
    }

    void initialize_data(int m, int n, int row_nnz)
    {
        b = gen_vec(m, n);
        x = gen_vec(m, n);
        mtx = gen_mtx(m, row_nnz);
        mtx_u = gen_u_mtx(m, row_nnz);
        dx = gko::clone(exec, x);
        db = gko::clone(exec, b);
        dmtx = gko::clone(exec, mtx);
        dmtx_u = gko::clone(exec, mtx_u);
    }

    std::shared_ptr<vec_type> b;
    std::shared_ptr<vec_type> x;
    std::shared_ptr<mtx_type> mtx;
    std::shared_ptr<mtx_type> mtx_u;
    std::shared_ptr<vec_type> db;
    std::shared_ptr<vec_type> dx;
    std::shared_ptr<mtx_type> dmtx;
    std::shared_ptr<mtx_type> dmtx_u;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::default_random_engine rand_engine;
};


TEST_F(UpperTrs, ApplyFullDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 4, 50);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 5, 50);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 6, 5);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyFullSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 7, 5);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 8, 50);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 9, 50);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 10, 5);
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ApplyTriangularSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 11, 5);
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


#ifdef GKO_COMPILING_CUDA


TEST_F(UpperTrs, ClassicalApplyFullDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 4, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 5, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyFullSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 6, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs,
       ClassicalApplyFullSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 7, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx);
    auto d_solver = d_upper_trs_factory->generate(dmtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 8, 50);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs,
       ClassicalApplyTriangularDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 9, 50);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs, ClassicalApplyTriangularSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 10, 5);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory = solver_type::build().on(ref);
    auto d_upper_trs_factory = solver_type::build().on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(UpperTrs,
       ClassicalApplyTriangularSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 11, 5);
    dmtx_u->set_strategy(std::make_shared<mtx_type::classical>());
    auto upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_upper_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = upper_trs_factory->generate(mtx_u);
    auto d_solver = d_upper_trs_factory->generate(dmtx_u);

    solver->apply(b.get(), x.get());
    d_solver->apply(db.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


#endif
