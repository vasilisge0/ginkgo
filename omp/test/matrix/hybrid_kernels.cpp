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

#include <ginkgo/core/matrix/hybrid.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/hybrid_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Hybrid : public ::testing::Test {
protected:
    using value_type = double;
    using Mtx = gko::matrix::Hybrid<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;

    Hybrid() : rand_engine(42) {}

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

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gen_mtx<MtxType>(num_rows, num_cols, min_nnz_row, num_cols);
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row, int max_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, max_nnz_row),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1,
                           std::shared_ptr<Mtx::strategy_type> strategy =
                               std::make_shared<Mtx::automatic>())
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx(532, 231, 1));
        expected = gen_mtx(532, num_vectors, 1);
        y = gen_mtx(231, num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(omp, strategy);
        dmtx->copy_from(mtx.get());
        dresult = gko::clone(omp, expected);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Hybrid, SubMatrixExecutorAfterCopyIsEquivalentToExcutor)
{
    set_up_apply_data();

    auto coo_mtx = dmtx->get_coo();
    auto ell_mtx = dmtx->get_ell();

    ASSERT_EQ(coo_mtx->get_executor(), omp);
    ASSERT_EQ(ell_mtx->get_executor(), omp);
    ASSERT_EQ(dmtx->get_executor(), omp);
}


TEST_F(Hybrid, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    mtx->apply(complex_b.get(), complex_x.get());
    dmtx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Hybrid, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = gko::clone(omp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = gko::clone(omp, complex_x);

    mtx->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Hybrid, ConvertEmptyCooToCsrIsEquivalentToRef)
{
    auto balanced_mtx =
        Mtx::create(ref, std::make_shared<Mtx::column_limit>(4));
    balanced_mtx->copy_from(gen_mtx(400, 200, 4, 4).get());
    auto dbalanced_mtx =
        Mtx::create(omp, std::make_shared<Mtx::column_limit>(4));
    dbalanced_mtx->copy_from(balanced_mtx.get());
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    balanced_mtx->convert_to(csr_mtx.get());
    dbalanced_mtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Hybrid, ConvertWithEmptyFirstAndLastRowToCsrIsEquivalentToRef)
{
    // create a dense matrix for easier manipulation
    auto dense_mtx = gen_mtx(400, 200, 0, 4);
    // set first and last row to zero
    for (gko::size_type col = 0; col < dense_mtx->get_size()[1]; col++) {
        dense_mtx->at(0, col) = gko::zero<value_type>();
        dense_mtx->at(dense_mtx->get_size()[0] - 1, col) =
            gko::zero<value_type>();
    }
    // now convert them to hybrid matrices
    auto balanced_mtx = gko::clone(ref, dense_mtx);
    auto dbalanced_mtx = gko::clone(omp, balanced_mtx);
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    balanced_mtx->convert_to(csr_mtx.get());
    dbalanced_mtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Hybrid, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx->convert_to(csr_mtx.get());
    dmtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 0);
}


TEST_F(Hybrid, MoveToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx->move_to(csr_mtx.get());
    dmtx->move_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Hybrid, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Hybrid, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
}


TEST_F(Hybrid, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    using AbsMtx = gko::remove_complex<Mtx>;

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();
    auto abs_strategy = gko::as<AbsMtx::column_limit>(abs_mtx->get_strategy());
    auto dabs_strategy =
        gko::as<AbsMtx::column_limit>(dabs_mtx->get_strategy());

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(),
                  dabs_strategy->get_num_columns());
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(), 2);
}


}  // namespace
