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

#ifndef GKO_CORE_PRECONDITIONER_JACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_JACOBI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL(ValueType, IndexType)          \
    void find_blocks(std::shared_ptr<const DefaultExecutor> exec,            \
                     const matrix::Csr<ValueType, IndexType>* system_matrix, \
                     uint32 max_block_size, size_type& num_blocks,           \
                     array<IndexType>& block_pointers)

#define GKO_DECLARE_JACOBI_GENERATE_KERNEL(ValueType, IndexType)           \
    void generate(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Csr<ValueType, IndexType>* system_matrix,            \
        size_type num_blocks, uint32 max_block_size,                       \
        remove_complex<ValueType> accuracy,                                \
        const preconditioner::block_interleaved_storage_scheme<IndexType>& \
            storage_scheme,                                                \
        array<remove_complex<ValueType>>& conditioning,                    \
        array<precision_reduction>& block_precisions,                      \
        const array<IndexType>& block_pointers, array<ValueType>& blocks)

#define GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL(ValueType)          \
    void scalar_conj(std::shared_ptr<const DefaultExecutor> exec, \
                     const array<ValueType>& diag,                \
                     array<ValueType>& conj_diag)

#define GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL(ValueType)          \
    void invert_diagonal(std::shared_ptr<const DefaultExecutor> exec, \
                         const array<ValueType>& diag,                \
                         array<ValueType>& inv_diag)

#define GKO_DECLARE_JACOBI_APPLY_KERNEL(ValueType, IndexType)                  \
    void apply(                                                                \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,     \
        uint32 max_block_size,                                                 \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&     \
            storage_scheme,                                                    \
        const array<precision_reduction>& block_precisions,                    \
        const array<IndexType>& block_pointers,                                \
        const array<ValueType>& blocks, const matrix::Dense<ValueType>* alpha, \
        const matrix::Dense<ValueType>* b,                                     \
        const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* x)

#define GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL(ValueType)          \
    void simple_scalar_apply(std::shared_ptr<const DefaultExecutor> exec, \
                             const array<ValueType>& diag,                \
                             const matrix::Dense<ValueType>* b,           \
                             matrix::Dense<ValueType>* x)

#define GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL(ValueType, IndexType)       \
    void simple_apply(                                                     \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks, \
        uint32 max_block_size,                                             \
        const preconditioner::block_interleaved_storage_scheme<IndexType>& \
            storage_scheme,                                                \
        const array<precision_reduction>& block_precisions,                \
        const array<IndexType>& block_pointers,                            \
        const array<ValueType>& blocks, const matrix::Dense<ValueType>* b, \
        matrix::Dense<ValueType>* x)

#define GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL(ValueType)                    \
    void scalar_apply(                                                       \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const array<ValueType>& diag, const matrix::Dense<ValueType>* alpha, \
        const matrix::Dense<ValueType>* b,                                   \
        const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* x)

#define GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL(ValueType, IndexType)          \
    void transpose_jacobi(                                                 \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks, \
        uint32 max_block_size,                                             \
        const array<precision_reduction>& block_precisions,                \
        const array<IndexType>& block_pointers,                            \
        const array<ValueType>& blocks,                                    \
        const preconditioner::block_interleaved_storage_scheme<IndexType>& \
            storage_scheme,                                                \
        array<ValueType>& out_blocks)

#define GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)     \
    void conj_transpose_jacobi(                                            \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks, \
        uint32 max_block_size,                                             \
        const array<precision_reduction>& block_precisions,                \
        const array<IndexType>& block_pointers,                            \
        const array<ValueType>& blocks,                                    \
        const preconditioner::block_interleaved_storage_scheme<IndexType>& \
            storage_scheme,                                                \
        array<ValueType>& out_blocks)

#define GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL(ValueType)          \
    void scalar_convert_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                                 const array<ValueType>& blocks,              \
                                 matrix::Dense<ValueType>* result)

#define GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType)   \
    void convert_to_dense(                                                 \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks, \
        const array<precision_reduction>& block_precisions,                \
        const array<IndexType>& block_pointers,                            \
        const array<ValueType>& blocks,                                    \
        const preconditioner::block_interleaved_storage_scheme<IndexType>& \
            storage_scheme,                                                \
        ValueType* result_values, size_type result_stride)

#define GKO_DECLARE_JACOBI_INITIALIZE_PRECISIONS_KERNEL                     \
    void initialize_precisions(std::shared_ptr<const DefaultExecutor> exec, \
                               const array<precision_reduction>& source,    \
                               array<precision_reduction>& precisions)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_GENERATE_KERNEL(ValueType, IndexType);         \
    template <typename ValueType>                                     \
    GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL(ValueType);                 \
    template <typename ValueType>                                     \
    GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL(ValueType);             \
    template <typename ValueType>                                     \
    GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL(ValueType);                \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_APPLY_KERNEL(ValueType, IndexType);            \
    template <typename ValueType>                                     \
    GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL(ValueType);         \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);   \
    template <typename ValueType>                                     \
    GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL(ValueType);     \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType); \
    GKO_DECLARE_JACOBI_INITIALIZE_PRECISIONS_KERNEL


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(jacobi, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_JACOBI_KERNELS_HPP_
