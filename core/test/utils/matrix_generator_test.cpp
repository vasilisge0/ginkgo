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

#include "core/test/utils/matrix_generator.hpp"


#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class MatrixGenerator : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using mtx_type = gko::matrix::Dense<T>;

    MatrixGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix<mtx_type>(
              500, 100, std::normal_distribution<real_type>(50, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          l_mtx(gko::test::generate_random_lower_triangular_matrix<mtx_type>(
              4, 3, true, std::normal_distribution<real_type>(50, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          u_mtx(gko::test::generate_random_upper_triangular_matrix<mtx_type>(
              3, 4, true, std::normal_distribution<real_type>(50, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          lower_bandwidth(2),
          upper_bandwidth(3),
          band_mtx(gko::test::generate_random_band_matrix<mtx_type>(
              100, lower_bandwidth, upper_bandwidth,
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          nnz_per_row_sample(500, 0),
          values_sample(0),
          band_values_sample(0)
    {
        // collect samples of nnz/row and values from the matrix
        for (int row = 0; row < mtx->get_size()[0]; ++row) {
            for (int col = 0; col < mtx->get_size()[1]; ++col) {
                auto val = mtx->at(row, col);
                if (val != gko::zero<T>()) {
                    ++nnz_per_row_sample[row];
                    values_sample.push_back(val);
                }
            }
        }

        // collect samples of values from the band matrix
        for (int row = 0; row < band_mtx->get_size()[0]; ++row) {
            for (int col = 0; col < band_mtx->get_size()[1]; ++col) {
                auto val = band_mtx->at(row, col);
                if ((col - row <= upper_bandwidth) &&
                    (row - col <= lower_bandwidth)) {
                    band_values_sample.push_back(val);
                }
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    int lower_bandwidth;
    int upper_bandwidth;
    std::unique_ptr<mtx_type> mtx;
    std::unique_ptr<mtx_type> l_mtx;
    std::unique_ptr<mtx_type> u_mtx;
    std::unique_ptr<mtx_type> band_mtx;
    std::vector<int> nnz_per_row_sample;
    std::vector<T> values_sample;
    std::vector<T> band_values_sample;


    template <typename InputIterator, typename ValueType, typename Closure>
    ValueType get_nth_moment(int n, ValueType c, InputIterator sample_start,
                             InputIterator sample_end, Closure closure_op)
    {
        using std::pow;
        ValueType res = 0;
        ValueType num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(closure_op(tmp) - c, n);
            num_elems += 1;
        }
        return res / num_elems;
    }

    template <typename ValueType, typename InputIterator, typename Closure>
    void check_average_and_deviation(
        InputIterator sample_start, InputIterator sample_end,
        gko::remove_complex<ValueType> average_ans,
        gko::remove_complex<ValueType> deviation_ans, Closure closure_op)
    {
        auto average =
            this->get_nth_moment(1, gko::zero<gko::remove_complex<ValueType>>(),
                                 sample_start, sample_end, closure_op);
        auto deviation = sqrt(this->get_nth_moment(2, average, sample_start,
                                                   sample_end, closure_op));

        // check that average & deviation is within 10% of the required amount
        ASSERT_NEAR(average, average_ans, average_ans * 0.1);
        ASSERT_NEAR(deviation, deviation_ans, deviation_ans * 0.1);
    }
};

TYPED_TEST_SUITE(MatrixGenerator, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MatrixGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(500, 100));
}


TYPED_TEST(MatrixGenerator, OutputHasCorrectNonzeroAverageAndDeviation)
{
    using T = typename TestFixture::value_type;
    // the nonzeros only needs to check the real part
    this->template check_average_and_deviation<T>(
        begin(this->nnz_per_row_sample), end(this->nnz_per_row_sample), 50.0,
        5.0, [](T val) { return gko::real(val); });
}


TYPED_TEST(MatrixGenerator, OutputHasCorrectValuesAverageAndDeviation)
{
    using T = typename TestFixture::value_type;
    // check the real part
    this->template check_average_and_deviation<T>(
        begin(this->values_sample), end(this->values_sample), 20.0, 5.0,
        [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            begin(this->values_sample), end(this->values_sample), 20.0, 5.0,
            [](T& val) { return gko::imag(val); });
    }
}


TYPED_TEST(MatrixGenerator, CanGenerateLowerTriangularMatrixWithDiagonalOnes)
{
    using T = typename TestFixture::value_type;
    ASSERT_EQ(this->l_mtx->at(0, 0), T{1.0});
    ASSERT_EQ(this->l_mtx->at(0, 1), T{0.0});
    ASSERT_EQ(this->l_mtx->at(0, 2), T{0.0});
    ASSERT_NE(this->l_mtx->at(1, 0), T{0.0});
    ASSERT_EQ(this->l_mtx->at(1, 1), T{1.0});
    ASSERT_EQ(this->l_mtx->at(1, 2), T{0.0});
    ASSERT_NE(this->l_mtx->at(2, 0), T{0.0});
    ASSERT_NE(this->l_mtx->at(2, 1), T{0.0});
    ASSERT_EQ(this->l_mtx->at(2, 2), T{1.0});
    ASSERT_NE(this->l_mtx->at(3, 0), T{0.0});
    ASSERT_NE(this->l_mtx->at(3, 1), T{0.0});
    ASSERT_NE(this->l_mtx->at(3, 2), T{0.0});
}


TYPED_TEST(MatrixGenerator, CanGenerateUpperTriangularMatrixWithDiagonalOnes)
{
    using T = typename TestFixture::value_type;
    ASSERT_EQ(this->u_mtx->at(0, 0), T{1.0});
    ASSERT_NE(this->u_mtx->at(0, 1), T{0.0});
    ASSERT_NE(this->u_mtx->at(0, 2), T{0.0});
    ASSERT_NE(this->u_mtx->at(0, 3), T{0.0});
    ASSERT_EQ(this->u_mtx->at(1, 0), T{0.0});
    ASSERT_EQ(this->u_mtx->at(1, 1), T{1.0});
    ASSERT_NE(this->u_mtx->at(1, 2), T{0.0});
    ASSERT_NE(this->u_mtx->at(1, 3), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 0), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 1), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 2), T{1.0});
    ASSERT_NE(this->u_mtx->at(2, 3), T{0.0});
}


TYPED_TEST(MatrixGenerator, CanGenerateBandMatrix)
{
    using T = typename TestFixture::value_type;
    // the elements out of band are zero
    for (int row = 0; row < this->band_mtx->get_size()[0]; row++) {
        for (int col = 0; col < this->band_mtx->get_size()[1]; col++) {
            if ((col - row > this->upper_bandwidth) ||
                (row - col > this->lower_bandwidth)) {
                ASSERT_EQ(this->band_mtx->at(row, col), T{0.0});
            }
        }
    }
    // check the real part of elements in band
    this->template check_average_and_deviation<T>(
        begin(this->band_values_sample), end(this->band_values_sample), 20.0,
        5.0, [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            begin(this->band_values_sample), end(this->band_values_sample),
            20.0, 5.0, [](T& val) { return gko::imag(val); });
    }
}


}  // namespace
