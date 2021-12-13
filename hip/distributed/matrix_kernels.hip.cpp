/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/distributed/matrix_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, global_index_type>>& input,
    const distributed::Partition<LocalIndexType>* partition,
    comm_index_type local_part,
    Array<matrix_data_entry<ValueType, LocalIndexType>>& diag_data,
    Array<matrix_data_entry<ValueType, LocalIndexType>>& offdiag_data,
    Array<LocalIndexType>& local_gather_idxs, comm_index_type* recv_offsets,
    Array<global_index_type>& local_row_to_global,
    Array<global_index_type>& local_offdiag_col_to_global,
    ValueType deduction_help) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BUILD_DIAG_OFFDIAG);


template <typename SourceType, typename TargetType>
void map_to_global_idxs(std::shared_ptr<const DefaultExecutor> exec,
                        const SourceType* input, size_t n, TargetType* output,
                        const TargetType* map) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAP_TO_GLOBAL_IDXS);


template <typename ValueType, typename LocalIndexType>
void merge_diag_offdiag(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, LocalIndexType>* diag,
                        const matrix::Csr<ValueType, LocalIndexType>* offdiag,
                        matrix::Csr<ValueType, LocalIndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MERGE_DIAG_OFFDIAG);

template <typename ValueType, typename LocalIndexType>
void combine_local_mtxs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, LocalIndexType>* local,
                        matrix::Csr<ValueType, LocalIndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMBINE_LOCAL_MTXS);


template <typename IndexType>
void build_recv_sizes(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* col_idxs, size_type num_cols,
                      const distributed::Partition<IndexType>* partition,
                      const global_index_type* map,
                      Array<comm_index_type>& recv_sizes,
                      Array<comm_index_type>& recv_offsets,
                      Array<IndexType>& recv_indices) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_BUILD_RECV_SIZES);


template <typename ValueType, typename LocalIndexType>
void compress_offdiag_data(
    std::shared_ptr<const DefaultExecutor> exec,
    device_matrix_data<ValueType, LocalIndexType>& offdiag_data,
    Array<global_index_type>& col_map) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPRESS_OFFDIAG_DATA);


}  // namespace distributed_matrix
}  // namespace hip
}  // namespace kernels
}  // namespace gko
