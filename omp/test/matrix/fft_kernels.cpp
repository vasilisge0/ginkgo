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

#include <ginkgo/core/matrix/fft.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class Fft : public ::testing::Test {
protected:
    using value_type = ValueType;
    using Vec = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::Fft;
    using Mtx2 = gko::matrix::Fft2;
    using Mtx3 = gko::matrix::Fft3;

    Fft()
        : ref{gko::ReferenceExecutor::create()},
          omp{gko::OmpExecutor::create()},
          rand_engine{9876567},
          n1{16},
          n2{32},
          n3{64},
          n{n1 * n2 * n3},
          cols{3},
          subcols{2},
          out_stride{6}
    {
        data = gko::test::generate_random_matrix<Vec>(
            n, cols, std::uniform_int_distribution<>(1, cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        ddata = Vec::create(omp);
        ddata->copy_from(this->data.get());
        data_strided = data->create_submatrix({0, n}, {0, subcols});
        ddata_strided = ddata->create_submatrix({0, n}, {0, subcols});
        out = data->clone();
        dout = data->clone();
        out_strided = Vec::create(ref, data_strided->get_size(), out_stride);
        dout_strided = Vec::create(omp, data_strided->get_size(), out_stride);
        fft = Mtx::create(ref, n);
        dfft = Mtx::create(omp, n);
        ifft = Mtx::create(ref, n, true);
        difft = Mtx::create(omp, n, true);
        fft2 = Mtx2::create(ref, n1 * n2, n3);
        dfft2 = Mtx2::create(omp, n1 * n2, n3);
        ifft2 = Mtx2::create(ref, n1 * n2, n3, true);
        difft2 = Mtx2::create(omp, n1 * n2, n3, true);
        fft3 = Mtx3::create(ref, n1, n2, n3);
        dfft3 = Mtx3::create(omp, n1, n2, n3);
        ifft3 = Mtx3::create(ref, n1, n2, n3, true);
        difft3 = Mtx3::create(omp, n1, n2, n3, true);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;
    std::default_random_engine rand_engine;
    size_t n1;
    size_t n2;
    size_t n3;
    size_t n;
    size_t cols;
    size_t subcols;
    size_t out_stride;
    std::unique_ptr<Vec> data;
    std::unique_ptr<Vec> ddata;
    std::unique_ptr<Vec> data_strided;
    std::unique_ptr<Vec> ddata_strided;
    std::unique_ptr<Vec> out;
    std::unique_ptr<Vec> dout;
    std::unique_ptr<Vec> out_strided;
    std::unique_ptr<Vec> dout_strided;
    std::unique_ptr<Mtx> fft;
    std::unique_ptr<Mtx> dfft;
    std::unique_ptr<Mtx> ifft;
    std::unique_ptr<Mtx> difft;
    std::unique_ptr<Mtx2> fft2;
    std::unique_ptr<Mtx2> dfft2;
    std::unique_ptr<Mtx2> ifft2;
    std::unique_ptr<Mtx2> difft2;
    std::unique_ptr<Mtx3> fft3;
    std::unique_ptr<Mtx3> dfft3;
    std::unique_ptr<Mtx3> ifft3;
    std::unique_ptr<Mtx3> difft3;
};


TYPED_TEST_SUITE(Fft, gko::test::ComplexValueTypes, TypenameNameGenerator);


TYPED_TEST(Fft, Apply1DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft->apply(this->data.get(), this->out.get());
    this->dfft->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided1DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft->apply(this->data_strided.get(), this->out_strided.get());
    this->dfft->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply1DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft->apply(this->data.get(), this->out.get());
    this->difft->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided1DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft->apply(this->data_strided.get(), this->out_strided.get());
    this->difft->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply2DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft2->apply(this->data.get(), this->out.get());
    this->dfft2->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided2DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft2->apply(this->data_strided.get(), this->out_strided.get());
    this->dfft2->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply2DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft2->apply(this->data.get(), this->out.get());
    this->difft2->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided2DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft2->apply(this->data_strided.get(), this->out_strided.get());
    this->difft2->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply3DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft3->apply(this->data.get(), this->out.get());
    this->dfft3->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided3DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft3->apply(this->data_strided.get(), this->out_strided.get());
    this->dfft3->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply3DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft3->apply(this->data.get(), this->out.get());
    this->difft3->apply(this->ddata.get(), this->dout.get());

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided3DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft3->apply(this->data_strided.get(), this->out_strided.get());
    this->difft3->apply(this->ddata_strided.get(), this->dout_strided.get());

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


}  // namespace
