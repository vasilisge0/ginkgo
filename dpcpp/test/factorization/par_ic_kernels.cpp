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

#include "core/factorization/par_ic_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


class ParIc : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = gko::int32;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    ParIc()
        : mtx_size(624, 624),
          rand_engine(43456),
          ref(gko::ReferenceExecutor::create()),
          dpcpp(gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create()))
    {
        mtx_l = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(0, 10.0),
            rand_engine, ref);

        dmtx_ani = Csr::create(dpcpp);
        dmtx_l_ani = Csr::create(dpcpp);
        dmtx_l_ani_init = Csr::create(dpcpp);
        dmtx_l = gko::clone(dpcpp, mtx_l);
    }

    void SetUp()
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        mtx_ani = gko::read<Csr>(input_file, ref);
        mtx_ani->sort_by_column_index();

        {
            mtx_l_ani = Csr::create(ref, mtx_ani->get_size());
            gko::matrix::CsrBuilder<value_type, index_type> l_builder(
                lend(mtx_l_ani));
            gko::kernels::reference::factorization::initialize_row_ptrs_l(
                ref, lend(mtx_ani), mtx_l_ani->get_row_ptrs());
            auto l_nnz =
                mtx_l_ani->get_const_row_ptrs()[mtx_ani->get_size()[0]];
            l_builder.get_col_idx_array().resize_and_reset(l_nnz);
            l_builder.get_value_array().resize_and_reset(l_nnz);
            gko::kernels::reference::factorization::initialize_l(
                ref, lend(mtx_ani), lend(mtx_l_ani), false);
            mtx_l_ani_init = gko::clone(ref, mtx_l_ani);
            gko::kernels::reference::par_ic_factorization::init_factor(
                ref, lend(mtx_l_ani_init));
        }
        dmtx_ani->copy_from(lend(mtx_ani));
        dmtx_l_ani->copy_from(lend(mtx_l_ani));
        dmtx_l_ani_init->copy_from(lend(mtx_l_ani_init));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::DpcppExecutor> dpcpp;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_ani;
    std::unique_ptr<Csr> mtx_l_ani;
    std::unique_ptr<Csr> mtx_l_ani_init;

    std::unique_ptr<Csr> dmtx_l;
    std::unique_ptr<Csr> dmtx_ani;
    std::unique_ptr<Csr> dmtx_l_ani;
    std::unique_ptr<Csr> dmtx_l_ani_init;
};


TEST_F(ParIc, KernelInitFactorIsEquivalentToRef)
{
    gko::kernels::reference::par_ic_factorization::init_factor(ref,
                                                               lend(mtx_l));
    gko::kernels::dpcpp::par_ic_factorization::init_factor(dpcpp, lend(dmtx_l));

    GKO_ASSERT_MTX_NEAR(mtx_l, dmtx_l, r<value_type>::value);
}


TEST_F(ParIc, KernelComputeFactorIsEquivalentToRef)
{
    auto square_size = mtx_ani->get_size();
    auto mtx_l_coo = Coo::create(ref, square_size);
    mtx_l_ani->convert_to(lend(mtx_l_coo));
    auto dmtx_l_coo = gko::clone(dpcpp, mtx_l_coo);

    gko::kernels::reference::par_ic_factorization::compute_factor(
        ref, 1, lend(mtx_l_coo), lend(mtx_l_ani_init));
    gko::kernels::dpcpp::par_ic_factorization::compute_factor(
        dpcpp, 100, lend(dmtx_l_coo), lend(dmtx_l_ani_init));

    GKO_ASSERT_MTX_NEAR(mtx_l_ani_init, dmtx_l_ani_init, 1e-4);
}


}  // namespace
