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

#include "core/base/index_set_kernels.hpp"


#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Omp namespace.
 *
 * @ingroup omp
 */
namespace omp {
/**
 * @brief The index_set namespace.
 *
 * @ingroup index_set
 */
namespace index_set {


template <typename IndexType>
void to_global_indices(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType num_subsets,
                       const IndexType* subset_begin,
                       const IndexType* subset_end,
                       const IndexType* superset_indices,
                       IndexType* decomp_indices)
{
#pragma omp parallel for
    for (size_type subset = 0; subset < num_subsets; ++subset) {
        for (size_type i = 0;
             i < superset_indices[subset + 1] - superset_indices[subset]; ++i) {
            decomp_indices[superset_indices[subset] + i] =
                subset_begin[subset] + i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL);


template <typename IndexType>
void populate_subsets(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType index_space_size,
                      const Array<IndexType>* indices,
                      Array<IndexType>* subset_begin,
                      Array<IndexType>* subset_end,
                      Array<IndexType>* superset_indices, const bool is_sorted)
{
    auto num_indices = indices->get_num_elems();
    auto tmp_indices = gko::Array<IndexType>(*indices);
    // Sort the indices if not sorted.
    if (!is_sorted) {
        std::sort(tmp_indices.get_data(), tmp_indices.get_data() + num_indices);
    }
    GKO_ASSERT(tmp_indices.get_const_data()[num_indices - 1] <=
               index_space_size);

    // Detect subsets.
    auto tmp_subset_begin = gko::vector<IndexType>(exec);
    auto tmp_subset_end = gko::vector<IndexType>(exec);
    auto tmp_subset_superset_index = gko::vector<IndexType>(exec);
    tmp_subset_begin.push_back(tmp_indices.get_data()[0]);
    tmp_subset_superset_index.push_back(0);
    for (size_type i = 1; i < num_indices; ++i) {
        if ((tmp_indices.get_data()[i] ==
             (tmp_indices.get_data()[i - 1] + 1)) ||
            (tmp_indices.get_data()[i] == tmp_indices.get_data()[i - 1])) {
            continue;
        }
        tmp_subset_end.push_back(tmp_indices.get_data()[i - 1] + 1);
        tmp_subset_superset_index.push_back(tmp_subset_superset_index.back() +
                                            tmp_subset_end.back() -
                                            tmp_subset_begin.back());
        tmp_subset_begin.push_back(tmp_indices.get_data()[i]);
    }
    tmp_subset_end.push_back(tmp_indices.get_data()[num_indices - 1] + 1);
    tmp_subset_superset_index.push_back(tmp_subset_superset_index.back() +
                                        tmp_subset_end.back() -
                                        tmp_subset_begin.back());

    // Make sure the sizes of the indices match and move them to their final
    // arrays.
    GKO_ASSERT(tmp_subset_begin.size() == tmp_subset_end.size());
    GKO_ASSERT((tmp_subset_begin.size() + 1) ==
               tmp_subset_superset_index.size());
    *subset_begin = std::move(gko::Array<IndexType>(
        exec, tmp_subset_begin.data(),
        tmp_subset_begin.data() + tmp_subset_begin.size()));
    *subset_end = std::move(
        gko::Array<IndexType>(exec, tmp_subset_end.data(),
                              tmp_subset_end.data() + tmp_subset_end.size()));
    *superset_indices = std::move(gko::Array<IndexType>(
        exec, tmp_subset_superset_index.data(),
        tmp_subset_superset_index.data() + tmp_subset_superset_index.size()));
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET_POPULATE_KERNEL);


template <typename IndexType>
void global_to_local(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType index_space_size,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* subset_end,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* global_indices, IndexType* local_indices,
                     const bool is_sorted)
{
#pragma omp parallel for
    for (size_type i = 0; i < num_indices; ++i) {
        auto index = global_indices[i];
        if (index >= index_space_size) {
            local_indices[i] = invalid_index<IndexType>();
            continue;
        }
        const auto bucket = std::distance(
            subset_begin,
            std::upper_bound(subset_begin, subset_begin + num_subsets, index));
        auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        if (subset_end[shifted_bucket] <= index) {
            local_indices[i] = invalid_index<IndexType>();
        } else {
            local_indices[i] = index - subset_begin[shifted_bucket] +
                               superset_indices[shifted_bucket];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL);


template <typename IndexType>
void local_to_global(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* subset_end,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* local_indices, IndexType* global_indices,
                     const bool is_sorted)
{
#pragma omp parallel for
    for (size_type i = 0; i < num_indices; ++i) {
        auto index = local_indices[i];
        if (index >= superset_indices[num_subsets]) {
            global_indices[i] = invalid_index<IndexType>();
            continue;
        }
        const auto bucket = std::distance(
            superset_indices,
            std::upper_bound(superset_indices,
                             superset_indices + num_subsets + 1, index));
        auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        global_indices[i] = subset_begin[shifted_bucket] + index -
                            superset_indices[shifted_bucket];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL);


}  // namespace index_set
}  // namespace omp
}  // namespace kernels
}  // namespace gko
