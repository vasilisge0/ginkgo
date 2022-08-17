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

#include <ginkgo/core/matrix/csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {
namespace arrow {
namespace {}  // anonymous namespace
}  // namespace arrow

template <typename ValueType, typename IndexType>
Arrow<ValueType, IndexType>& Arrow<ValueType, IndexType>::operator=(
    const Arrow<ValueType, IndexType>& other)
{
    if (&other != this) {
        EnableLinOp<Arrow>::operator=(other);
        this->submtx_00_ = other.submtx_00_;
        this->submtx_01_ = other.submtx_01_;
        this->submtx_10_ = other.submtx_10_;
        this->submtx_11_ = other.submtx_11_;
    }
    return *this;
}

template <typename ValueType, typename IndexType>
Arrow<ValueType, IndexType>& Arrow<ValueType, IndexType>::operator=(
    Arrow<ValueType, IndexType>&& other)
{
    if (&other != this) {
        EnableLinOp<Arrow>::operator=(other);
        this->submtx_00_ = std::move(other.submtx_00_);
        this->submtx_01_ = std::move(other.submtx_01_);
        this->submtx_10_ = std::move(other.submtx_10_);
        this->submtx_11_ = std::move(other.submtx_11_);
    }
    return *this;
}


}  // namespace matrix
}  // namespace gko