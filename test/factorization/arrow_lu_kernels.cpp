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

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/arrow.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>



#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {

template <typename ValueIndexType>
class Arrow_Lu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using partition_type = gko::array<index_type>;
    using Mtx = gko::matrix::Dense<value_type>;

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        dtmp.set_executor(ref);
        dtmp.set_executor(exec);
        matrices.emplace_back(new gko::array<index_type>(ref, {0, 3, 6, 8}),
                              gko::initialize<matrix_type>(
                                             {{2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0},
                                              {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                                              {0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0},
                                              {0.0, 0.0, 0.0, 3.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.2},
                                              {0.0, 0.0, 0.0,-1.0, 3.0,-1.0, 0.0, 0.0, 0.1, 0.0},
                                              {0.0, 0.0, 0.0, 0.0,-1.0, 3.0, 0.0, 0.0, 0.0, 0.0},
                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,-1.0, 0.0, 0.0},
                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 2.0, 0.1, 0.0},
                                              {0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 2.0, 0.1},
                                              {0.0, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 2.0}}, ref));
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    static void assert_same_matrices(const Mtx* m1, const Mtx* m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }

    static void assert_same_matrices(const matrix_type* m1, const matrix_type* m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        ASSERT_EQ(m1->get_num_stored_elements(), m2->get_num_stored_elements());
        for (gko::size_type i = 0; i < m1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m1->get_const_values()[i], m2->get_const_values()[i]);
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::vector<std::pair<std::unique_ptr<gko::array<index_type>>, std::shared_ptr<matrix_type>>>
        matrices;
    gko::array<index_type> tmp;
    gko::array<index_type> dtmp;
};

using Types = ::testing::Types<std::tuple<float, gko::int32>>;

TYPED_TEST_SUITE(Arrow_Lu, Types, PairTypenameNameGenerator);

TYPED_TEST(Arrow_Lu, KernelTest)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto ref_exec = gko::ReferenceExecutor::create();
    for (auto& pair : this->matrices) {
        //SCOPED_TRACE(pair.first);
        auto& partition_idxs = pair.first;
        //auto& mtx = pair.second;
        auto A = pair.second;
        auto dmtx = gko::clone(ref_exec, A);
        auto idx = partition_idxs->get_num_elems()-1;
        auto split_index = partition_idxs->get_data()[idx];
        gko::array<index_type> partitions = {ref_exec, partition_idxs->get_num_elems()};
        for (auto i = 0; i < partition_idxs->get_num_elems(); i++) {
            partitions.get_data()[i] = partition_idxs->get_data()[i];
        }
        // auto lu_fact = gko::factorization::ArrowLu<value_type, index_type>(A);
        auto lu_fact =
            gko::factorization::ArrowLu<value_type, index_type>::build().on(ref_exec);
        // std::shared_ptr<gko::matrix::Arrow<value_type, index_type>> arrow_mtx;
        // auto arrow_mtx = gko::matrix::Csr<value_type, index_type>::create();
        auto arrow_mtx = share(gko::matrix::Arrow<value_type, index_type>::create(ref_exec, *partition_idxs.get()));
        
        // auto arrow_mtx = gko::matrix::Csr<value_type, index_type>::create(ref_exec);

        gko::as<gko::ConvertibleTo<gko::matrix::Arrow<value_type, index_type>>>(A.get())
            ->convert_to(arrow_mtx.get());

        auto D = arrow_mtx->get_submatrix_00().get();
        auto d0 = gko::as<gko::matrix::Dense<value_type>>(D->begin()[0].get());
        auto d1 = gko::as<gko::matrix::Dense<value_type>>(D->begin()[1].get());
        auto d2 = gko::as<gko::matrix::Dense<value_type>>(D->begin()[2].get());

        auto d0_true = gko::initialize<gko::matrix::Dense<value_type>>(
                                                         {{2.0, 0.0, 0.5},
                                                          {0.0, 2.0, 0.0},
                                                          {0.5, 0.0, 2.0}}, ref_exec);
        auto d1_true = gko::initialize<gko::matrix::Dense<value_type>>(
                                                         {{3.0,-1.0, 0.0},
                                                          {-1.0, 3.0,-1.0},
                                                          {0.0,-1.0, 3.0}}, ref_exec);
        auto d2_true = gko::initialize<gko::matrix::Dense<value_type>>(
                                                         {{ 2.0,-1.0},
                                                          {-1.0, 2.0}}, ref_exec);                                                  

        this->assert_same_matrices(d0, d0_true.get());
        this->assert_same_matrices(d1, d1_true.get());
        this->assert_same_matrices(d2, d2_true.get());                                                                                                                                                  

        auto E = arrow_mtx->get_submatrix_01().get();
        auto e0 = gko::as<gko::matrix::Csr<value_type, index_type>>(E->begin()[0].get());
        auto e1 = gko::as<gko::matrix::Csr<value_type, index_type>>(E->begin()[1].get());
        auto e2 = gko::as<gko::matrix::Csr<value_type, index_type>>(E->begin()[2].get());

        auto e0_true = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.2, 0.0},
                                                          {0.0, 1.0},
                                                          {0.1, 0.0}}, ref_exec);
        auto e1_true = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.0, 0.2},
                                                          {0.1, 0.0},
                                                          {0.0, 0.0}}, ref_exec);
        auto e2_true = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.0, 0.0},
                                                          {0.1, 0.0}}, ref_exec);
        this->assert_same_matrices(e0, e0_true.get());
        this->assert_same_matrices(e1, e1_true.get());
        this->assert_same_matrices(e2, e2_true.get());

        auto F = arrow_mtx->get_submatrix_10().get();
        auto f0 = gko::as<gko::matrix::Csr<value_type, index_type>>(F->begin()[0].get());
        auto f1 = gko::as<gko::matrix::Csr<value_type, index_type>>(F->begin()[1].get());
        auto f2 = gko::as<gko::matrix::Csr<value_type, index_type>>(F->begin()[2].get());

        auto f0_true_tmp = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.2, 0.0, 0.1,},
                                                          {0.0, 0.8, 0.0,}}, ref_exec);
        auto f0_true = gko::as<gko::matrix::Csr<value_type, index_type>>(f0_true_tmp->transpose());
        auto f1_true_tmp = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.0, 0.1, 0.0},
                                                          {0.2, 0.0, 0.0}}, ref_exec);
        auto f1_true = gko::as<gko::matrix::Csr<value_type, index_type>>(f1_true_tmp->transpose());
        auto f2_true_tmp = gko::initialize<gko::matrix::Csr<value_type, index_type>>(
                                                         {{0.1, 0.0},
                                                          {0.0, 0.1}}, ref_exec);
        auto f2_true = gko::as<gko::matrix::Csr<value_type, index_type>>(f2_true_tmp->transpose());

        this->assert_same_matrices(f0, f0_true.get());
        this->assert_same_matrices(f1, f1_true.get());
        this->assert_same_matrices(f2, f2_true.get());

        auto C = arrow_mtx->get_submatrix_11().get();
        auto c0 = gko::as<gko::matrix::Dense<value_type>>(C->begin()[0].get());
        auto c0_true = gko::initialize<gko::matrix::Dense<value_type>>(
                                                         {{2.0, 0.1},
                                                          {0.1, 2.0}}, ref_exec);
        this->assert_same_matrices(c0, c0_true.get());
        std::cout << "<<<before generate>>>\n";
        auto fact = gko::share(lu_fact->generate(arrow_mtx));
        std::cout << "after generate\n";
        std::cout << " --- printout --- " << '\n';
        
        // auto y = gko::as<gko::matrix::Dense<value_type>>(t0[0].get());
        // std::cout << "t0[0].get_values()[0]: " << gko::as<gko::matrix::Dense>(t0[0]).get_values()[0] << '\n';
        // std::cout << "arrow_mtx->get_submtx_00().get_values()[0]: " << arrow_mtx->get_submatrix_00()[0].get_values()[0] << '\n';
        //GKO_ASSERT_ARRAY_EQ(drow_nnz, row_nnz);
    }
}

}