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

#include "core/preconditioner/jacobi_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/warp_blas.cuh"
#include "cuda/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


#include "common/cuda_hip/preconditioner/jacobi_advanced_apply_kernel.hpp.inc"


// clang-format off
#cmakedefine GKO_JACOBI_BLOCK_SIZE @GKO_JACOBI_BLOCK_SIZE@
// clang-format on
// make things easier for IDEs
#ifndef GKO_JACOBI_BLOCK_SIZE
#define GKO_JACOBI_BLOCK_SIZE 1
#endif


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void advanced_apply(
    syn::value_list<int, max_block_size>, size_type num_blocks,
    const precision_reduction* block_precisions,
    const IndexType* block_pointers, const ValueType* blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const ValueType* alpha, const ValueType* b, size_type b_stride,
    ValueType* x, size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (grid_size > 0) {
        if (block_precisions) {
            kernel::advanced_adaptive_apply<max_block_size, subwarp_size,
                                            warps_per_block>
                <<<grid_size, block_size, 0, 0>>>(
                    as_cuda_type(blocks), storage_scheme, block_precisions,
                    block_pointers, num_blocks, as_cuda_type(alpha),
                    as_cuda_type(b), b_stride, as_cuda_type(x), x_stride);
        } else {
            kernel::advanced_apply<max_block_size, subwarp_size,
                                   warps_per_block>
                <<<grid_size, block_size, 0, 0>>>(
                    as_cuda_type(blocks), storage_scheme, block_pointers,
                    num_blocks, as_cuda_type(alpha), as_cuda_type(b), b_stride,
                    as_cuda_type(x), x_stride);
        }
    }
}


#define DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION(ValueType, IndexType)   \
    void advanced_apply<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE, \
                        ValueType, IndexType>(                              \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>, size_type,             \
        const precision_reduction*, const IndexType* block_pointers,        \
        const ValueType*,                                                   \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&, \
        const ValueType*, const ValueType*, size_type, ValueType*, size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION);


}  // namespace jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
