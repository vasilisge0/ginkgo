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


#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"
#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {

template <typename ValueIndexType>
class ArrowLu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using partition_type = gko::array<index_type>;

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
                                              {0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 2.0, 0.1},
                                              {0.0, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 2.0}}, ref));
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
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

TYPED_TEST_SUITE(ArrowLu, Types, PairTypenameNameGenerator);

TYPED_TEST(ArrowLu, KernelTest)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto ref_exec = gko::ReferenceExecutor::create();
    for (auto& pair : this->matrices) {
        //SCOPED_TRACE(pair.first);
        auto& partition_idxs = pair.first;
        //auto& mtx = pair.second;
        std::cout << "testing\n";
        auto A = pair.second;
        auto dmtx = gko::clone(ref_exec, A);
        auto idx = partition_idxs->get_num_elems()-1;
        auto split_index = partition_idxs->get_data()[idx];
        gko::array<index_type> partitions = {ref_exec, partition_idxs->get_num_elems()};
        for (auto i = 0; i < partition_idxs->get_num_elems(); i++) {
            partitions.get_data()[i] = partition_idxs->get_data()[i];
        }
        // auto lu_fact =
        // gko::factorization::ArrowLu<value_type, index_type>::build()
        //     .with_workspace(std::move(new gko::factorization::arrow_lu_workspace(A, partitions, split_index)))
        //     .on(this->exec);
        // auto fact = gko::share(lu_fact->generate(A));

        auto workspace = new gko::factorization::arrow_lu_workspace(A, partitions, split_index);
        gko::kernels::reference::arrow_lu::compute_factors(ref_exec, workspace, A.get());
        // std::cout << "split_index: " << fact->parameters_.workspace->get_submatrix_11()->split_index << '\n';
        // gko::kernels::reference::arrow_lu::factorize_submatrix_11<value_type, index_type>(mtx.get(), submtx_11, partitions);
        //gko::kernels::EXEC_NAMESPACE::arrow_lu::factorize_submatrix_11<value_type, index_type>(mtx.get(), submtx_11, partitions);

        //auto forest = gko::factorization::compute_elim_forest(mtx.get());
    //    //auto dforest = gko::factorization::compute_elim_forest(dmtx.get());
    //    //gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};
    //    //gko::array<index_type> drow_nnz{this->exec, mtx->get_size()[0]};

    //    //gko::kernels::reference::cholesky::cholesky_symbolic_count(
    //    //    this->ref, mtx.get(), forest, row_nnz.get_data(), this->tmp);
    //    //gko::kernels::EXEC_NAMESPACE::cholesky::cholesky_symbolic_count(
    //    //    this->exec, dmtx.get(), dforest, drow_nnz.get_data(), this->dtmp);

        //GKO_ASSERT_ARRAY_EQ(drow_nnz, row_nnz);
    }
}

}