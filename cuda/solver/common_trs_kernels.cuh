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

#ifndef GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
#define GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_


#include <functional>
#include <iostream>
#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual ~SolveStruct() = default;
};


}  // namespace solver


namespace kernels {
namespace cuda {
namespace {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    cusparseHandle_t handle;
    cusparseSpSMDescr_t spsm_descr;
    cusparseSpMatDescr_t descr_a;

    // Implicit parameter in spsm_solve, therefore stored here.
    array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper)
        : handle{exec->get_cusparse_handle()},
          spsm_descr{},
          descr_a{},
          work{exec}
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        spsm_descr = cusparse::create_spsm_descr();
        descr_a = cusparse::create_csr(
            matrix->get_size()[0], matrix->get_size()[1],
            matrix->get_num_stored_elements(),
            const_cast<IndexType*>(matrix->get_const_row_ptrs()),
            const_cast<IndexType*>(matrix->get_const_col_idxs()),
            const_cast<ValueType*>(matrix->get_const_values()));
        cusparse::set_attribute<cusparseFillMode_t>(
            descr_a, CUSPARSE_SPMAT_FILL_MODE,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        cusparse::set_attribute<cusparseDiagType_t>(
            descr_a, CUSPARSE_SPMAT_DIAG_TYPE, CUSPARSE_DIAG_TYPE_NON_UNIT);

        const auto rows = matrix->get_size()[0];
        // workaround suggested by NVIDIA engineers: for some reason
        // cusparse needs non-nullptr input vectors even for analysis
        auto descr_b = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAD));
        auto descr_c = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAF));

        auto work_size = cusparse::spsm_buffer_size(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(), descr_a,
            descr_b, descr_c, CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        work.resize_and_reset(work_size);

        cusparse::spsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                one<ValueType>(), descr_a, descr_b, descr_c,
                                CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr,
                                work.get_data());

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    void solve(const matrix::Csr<ValueType, IndexType>*,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        auto descr_b = cusparse::create_dnmat(
            input->get_size(), input->get_stride(),
            const_cast<ValueType*>(input->get_const_values()));
        auto descr_c = cusparse::create_dnmat(
            output->get_size(), output->get_stride(), output->get_values());

        cusparse::spsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(),
                             descr_a, descr_b, descr_c,
                             CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    ~CudaSolveStruct()
    {
        if (descr_a) {
            cusparse::destroy(descr_a);
            descr_a = nullptr;
        }
        if (spsm_descr) {
            cusparse::destroy(spsm_descr);
            spsm_descr = nullptr;
        }
    }

    CudaSolveStruct(const SolveStruct&) = delete;

    CudaSolveStruct(SolveStruct&&) = delete;

    CudaSolveStruct& operator=(const SolveStruct&) = delete;

    CudaSolveStruct& operator=(SolveStruct&&) = delete;
};


#elif (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))

template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    std::shared_ptr<const gko::CudaExecutor> exec;
    cusparseHandle_t handle;
    int algorithm;
    csrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    mutable array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper)
        : exec{exec},
          handle{exec->get_cusparse_handle()},
          algorithm{},
          solve_info{},
          policy{},
          factor_descr{},
          work{exec}
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        factor_descr = cusparse::create_mat_descr();
        solve_info = cusparse::create_solve_info();
        cusparse::set_mat_fill_mode(
            factor_descr,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        algorithm = 0;
        policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        size_type work_size{};

        cusparse::buffer_size_ext(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            &work_size);

        // allocate workspace
        work.resize_and_reset(work_size);

        cusparse::csrsm2_analysis(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            work.get_data());
    }

    void solve(const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        dense::copy(exec, input, output);
        cusparse::csrsm2_solve(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
            output->get_stride(), matrix->get_num_stored_elements(),
            one<ValueType>(), factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            output->get_values(), output->get_stride(), solve_info, policy,
            work.get_data());
    }

    ~CudaSolveStruct()
    {
        if (factor_descr) {
            cusparse::destroy(factor_descr);
            factor_descr = nullptr;
        }
        if (solve_info) {
            cusparse::destroy(solve_info);
            solve_info = nullptr;
        }
    }

    CudaSolveStruct(const CudaSolveStruct&) = delete;

    CudaSolveStruct(CudaSolveStruct&&) = delete;

    CudaSolveStruct& operator=(const CudaSolveStruct&) = delete;

    CudaSolveStruct& operator=(CudaSolveStruct&&) = delete;
};


#endif


void should_perform_transpose_kernel(std::shared_ptr<const CudaExecutor> exec,
                                     bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        solve_struct = std::make_shared<CudaSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* matrix,
                  const solver::SolveStruct* solve_struct,
                  matrix::Dense<ValueType>* trans_b,
                  matrix::Dense<ValueType>* trans_x,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* x)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    using vec = matrix::Dense<ValueType>;

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<const CudaSolveStruct<ValueType, IndexType>*>(
                    solve_struct)) {
            cuda_solve_struct->solve(matrix, b, x, trans_b, trans_x);
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


constexpr int default_block_size = 512;
constexpr int fallback_block_size = 32;


template <typename ValueType, typename IndexType>
__device__ __forceinline__
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType>
    load(const ValueType* values, IndexType index)
{
    const volatile ValueType* val = values + index;
    return *val;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ std::enable_if_t<
    std::is_floating_point<ValueType>::value, thrust::complex<ValueType>>
load(const thrust::complex<ValueType>* values, IndexType index)
{
    auto real = reinterpret_cast<const ValueType*>(values);
    auto imag = real + 1;
    return {load(real, 2 * index), load(imag, 2 * index)};
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(
    ValueType* values, IndexType index,
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType> value)
{
    volatile ValueType* val = values + index;
    *val = value;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(thrust::complex<ValueType>* values,
                                      IndexType index,
                                      thrust::complex<ValueType> value)
{
    auto real = reinterpret_cast<ValueType*>(values);
    auto imag = real + 1;
    store(real, 2 * index, value.real());
    store(imag, 2 * index, value.imag());
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool* nan_produced, IndexType* atomic_counter)
{
    __shared__ UninitializedArray<ValueType, default_block_size> x_s_array;
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_diag = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    ValueType sum = 0.0;
    for (auto i = row_begin; i != row_diag; i += row_step) {
        const auto dependency = colidxs[i];
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load(x_p, 0);
        }

        sum += x * vals[i];
    }

    const auto r = (b[row * b_stride + rhs] - sum) / vals[row_diag];

    store(x_s, self_shid, r);
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store(x_s, self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_legacy_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool* nan_produced, IndexType* atomic_counter)
{
    __shared__ IndexType block_base_idx;
    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * fallback_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_diag = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    ValueType sum = 0.0;
    auto j = row_begin;
    while (j != row_diag + row_step) {
        auto col = colidxs[j];
        auto x_val = load(x, col * x_stride + rhs);
        while (!is_nan(x_val)) {
            sum += vals[j] * x_val;
            j += row_step;
            col = colidxs[j];
            x_val = load(x, col * x_stride + rhs);
        }
        if (row == col) {
            const auto r = (b[row * b_stride + rhs] - sum) / vals[row_diag];
            store(x, row * x_stride + rhs, r);
            j += row_step;
            if (is_nan(r)) {
                store(x, row * x_stride + rhs, zero<ValueType>());
                *nan_produced = true;
            }
        }
    }
}


template <typename IndexType>
__global__ void sptrsv_init_kernel(bool* const nan_produced,
                                   IndexType* const atomic_counter)
{
    *nan_produced = false;
    *atomic_counter = IndexType{};
}


template <bool is_upper, typename ValueType, typename IndexType>
void sptrsv_naive_caching(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x)
{
    // Pre-Volta GPUs may deadlock due to missing independent thread scheduling.
    const auto is_fallback_required = exec->get_major_version() < 7;

    const auto n = matrix->get_size()[0];
    const auto nrhs = b->get_size()[1];

    // Initialize x to all NaNs.
    dense::fill(exec, x, nan<ValueType>());

    array<bool> nan_produced(exec, 1);
    array<IndexType> atomic_counter(exec, 1);
    sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                 atomic_counter.get_data());

    const dim3 block_size(
        is_fallback_required ? fallback_block_size : default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

    if (is_fallback_required) {
        sptrsv_naive_legacy_kernel<is_upper><<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
            nan_produced.get_data(), atomic_counter.get_data());
    } else {
        sptrsv_naive_caching_kernel<is_upper><<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
            nan_produced.get_data(), atomic_counter.get_data());
    }

#if GKO_VERBOSE_LEVEL >= 1
    if (exec->copy_val_to_host(nan_produced.get_const_data())) {
        std::cerr
            << "Error: triangular solve produced NaN, either not all diagonal "
               "elements are nonzero, or the system is very ill-conditioned. "
               "The NaN will be replaced with a zero.\n";
    }
#endif  // GKO_VERBOSE_LEVEL >= 1
}


}  // namespace
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
