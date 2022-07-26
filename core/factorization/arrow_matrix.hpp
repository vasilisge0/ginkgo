#ifndef ARROW_MATRIX_HPP
#define ARROW_MATRIX_HPP

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <vector>

// #include <ginkgo/core/factorization/arrow_lu.hpp>


namespace gko {
namespace factorization {

enum arrow_lu_type { arrow_lu_sequential, arrow_lu_omp, arrow_lu_cuda };

template <typename ValueType, typename IndexType>
struct arrow_matrix;

template <typename ValueType, typename IndexType>
struct arrow_submatrix_11;

template <typename ValueType, typename IndexType>
struct arrow_submatrix_12;

template <typename ValueType, typename IndexType>
struct arrow_submatrix_21;

template <typename ValueType, typename IndexType>
struct arrow_submatrix_22;

template <typename IndexType>
struct block_csr_storage {
    IndexType num_elems;
    IndexType num_blocks;
    array<IndexType> rows;
    array<IndexType> row_ptrs;
    array<IndexType> block_ptrs;

    block_csr_storage() {}

    block_csr_storage(std::shared_ptr<const Executor> exec,
                      IndexType num_elems_in, IndexType num_blocks_in)
    {
        num_elems = num_elems_in;
        num_blocks_in = num_blocks_in;
        array<IndexType> rows = {exec, num_elems};
        array<IndexType> row_ptrs = {exec, num_elems};
        array<IndexType> block_ptrs = {exec, num_blocks};
    }

    block_csr_storage(array<IndexType>& rows_in, array<IndexType>& row_ptrs_in,
                      array<IndexType>& block_ptrs_in)
    {
        num_elems = rows_in.get_num_elems();
        num_blocks = block_ptrs.get_num_elems();
        rows = std::move(rows_in);
        row_ptrs = std::move(row_ptrs_in);
        block_ptrs = std::move(block_ptrs_in);
    }

    block_csr_storage(std::shared_ptr<const Executor> exec,
                      IndexType num_elems_in, IndexType num_blocks_in,
                      IndexType* rows_in, IndexType* row_ptrs_in,
                      IndexType* block_row_ptrs_in)
    {
        num_elems = num_elems_in;
        num_blocks = num_blocks_in;
        rows = {exec, static_cast<size_type>(num_elems), rows_in};
        row_ptrs = {exec, static_cast<size_type>(num_elems), row_ptrs_in};
        row_ptrs = {exec, static_cast<size_type>(num_elems), row_ptrs_in};
        block_ptrs = {exec, static_cast<size_type>(num_blocks_in) + 1,
                      block_row_ptrs_in};
    }

    void reset_block_ptrs();

    void resize();
};


// Declaration of arrow_partitions struct.

template <typename IndexType>
struct arrow_partitions {
public:
    array<IndexType> data;
    IndexType split_index = 0;
    IndexType num_endpoints = 0;
    dim<2> size;

    arrow_partitions();
    arrow_partitions(arrow_partitions<IndexType>& in);
    arrow_partitions(std::shared_ptr<const Executor> exec,
                     std::ifstream& infile);
    arrow_partitions(gko::array<IndexType>& partition_idxs,
                     IndexType split_index_in);
    arrow_partitions(gko::array<IndexType>* partition_idxs,
                     IndexType split_index_in);
    arrow_partitions(std::unique_ptr<gko::array<IndexType>> partition_idxs,
                     IndexType split_index_in);
    arrow_partitions(const arrow_partitions& partitions_in);

    void read(std::ifstream& infile);
    IndexType* get_data();
    const IndexType* get_const_data();
    IndexType get_num_blocks();
};


// Declaration of arrow_matrix.

template <typename ValueType, typename IndexType>
struct arrow_matrix {
public:
    dim<2> size_;
    arrow_partitions<IndexType> partitions_;
    arrow_submatrix_11<ValueType, IndexType> submtx_11_;
    arrow_submatrix_12<ValueType, IndexType> submtx_12_;
    arrow_submatrix_21<ValueType, IndexType> submtx_21_;
    arrow_submatrix_22<ValueType, IndexType> submtx_22_;

    arrow_matrix(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 std::basic_ifstream<char, std::char_traits<char>>& infile);
    arrow_matrix(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 arrow_partitions<IndexType>& partitions);
    arrow_matrix(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 gko::array<IndexType>& partition_idxs,
                 IndexType split_index_in);

    dim<2> get_size();
};


// Declaration of arrow_submatrix_11 struct.

template <typename ValueType, typename IndexType>
struct arrow_submatrix_11 {
    using dense_mtx = matrix::Dense<ValueType>;
    using csr_mtx = matrix::Csr<ValueType, IndexType>;
    std::shared_ptr<const Executor> exec = nullptr;
    dim<2> size;
    IndexType nz = IndexType{};
    IndexType num_blocks = IndexType{};
    IndexType max_block_size = IndexType{};
    IndexType max_nz = IndexType{};
    IndexType nnz_l_factor = IndexType{};
    IndexType nnz_u_factor = IndexType{};
    IndexType split_index = IndexType{};
    array<IndexType> row_ptrs_tmp;
    std::vector<std::unique_ptr<dense_mtx>> dense_l_factors;
    std::vector<std::unique_ptr<dense_mtx>> dense_u_factors;
    std::vector<std::unique_ptr<dense_mtx>> dense_diagonal_blocks;
    std::vector<std::unique_ptr<csr_mtx>> sparse_diagonal_blocks;

    arrow_submatrix_11(
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
        arrow_partitions<IndexType>& arrow_partitions);
};


// Declaration of arrow_submatrix_12 struct.

template <typename ValueType, typename IndexType>
struct arrow_submatrix_12 {
    using csr_mtx = matrix::Csr<ValueType, IndexType>;
    std::shared_ptr<const Executor> exec = nullptr;
    dim<2> size;
    IndexType nz = IndexType{};
    IndexType num_blocks = IndexType{};
    IndexType max_block_size = IndexType{};
    IndexType max_nz = IndexType{};
    IndexType split_index;
    array<IndexType> row_ptrs;
    array<IndexType> block_row_ptrs;
    array<IndexType> row_ptrs_tmp;
    array<IndexType> row_ptrs_tmp2;
    array<IndexType> nz_per_block;
    std::shared_ptr<csr_mtx> mtx;

    arrow_submatrix_12(
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,
        arrow_partitions<IndexType>& partitions);
};


// Declaration of arrow_submatrix_21 struct.

template <typename ValueType, typename IndexType>
struct arrow_submatrix_21 {
    using csr_mtx = matrix::Csr<ValueType, IndexType>;
    std::shared_ptr<const Executor> exec = nullptr;
    dim<2> size;
    IndexType nz = IndexType{};
    IndexType num_blocks = IndexType{};
    IndexType max_block_size = IndexType{};
    IndexType max_nz = IndexType{};
    IndexType split_index;
    array<IndexType> row_ptrs_tmp;
    array<IndexType> row_ptrs_tmp2;
    array<IndexType> col_ptrs_tmp;
    array<IndexType> col_ptrs;
    std::shared_ptr<csr_mtx> mtx;
    array<IndexType> block_col_ptrs;
    array<IndexType> nz_per_block;
    std::shared_ptr<block_csr_storage<IndexType>> block_storage;

    arrow_submatrix_21(
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,
        arrow_partitions<IndexType>& partitions);
};


// Declaration of arrow_submatrix_21 struct.
template <typename ValueType, typename IndexType>
struct arrow_submatrix_22 {
    using dense_mtx = matrix::Dense<ValueType>;
    std::shared_ptr<const Executor> exec = nullptr;
    dim<2> size;
    IndexType split_index;
    std::shared_ptr<dense_mtx> schur_complement;
    std::shared_ptr<dense_mtx> dense_l_factor;
    std::shared_ptr<dense_mtx> dense_u_factor;

    arrow_submatrix_22(
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,
        arrow_submatrix_12<ValueType, IndexType>& submtx_12,
        arrow_submatrix_21<ValueType, IndexType>& submtx_21,
        arrow_partitions<IndexType>& partitions);

    arrow_submatrix_22(const arrow_submatrix_22<ValueType, IndexType>& input);
};

template <typename IndexType>
void arrow_partitions<IndexType>::read(std::ifstream& infile)
{
    // infile >> split_index;
    for (auto i = 0; i < size[0]; ++i) {
        // infile >> data.get_data()[i];
    }
}

template <typename IndexType>
IndexType* arrow_partitions<IndexType>::get_data()
{
    return this->data.get_data();
}

template <typename IndexType>
const IndexType* arrow_partitions<IndexType>::get_const_data()
{
    return this->data.get_const_data();
}

template <typename IndexType>
IndexType arrow_partitions<IndexType>::get_num_blocks()
{
    return this->num_endpoints - 1;
}

template <typename ValueType, typename IndexType>
struct arrow_lu_workspace {
    arrow_matrix<ValueType, IndexType> mtx_;

    arrow_lu_workspace(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                       arrow_partitions<IndexType>& partitions)
        : mtx_(mtx, partitions)
    {}
    arrow_lu_workspace(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                       std::ifstream& instream)
        : mtx_(mtx, instream)
    {}
    arrow_lu_workspace(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                       gko::array<IndexType>& partitions,
                       IndexType split_index_in)
        : mtx_(mtx, partitions, split_index_in)
    {}

    arrow_matrix<ValueType, IndexType>* get_mtx() {}
    arrow_partitions<IndexType>* get_partitions() {}
};

template <typename IndexType>
arrow_partitions<IndexType> compute_partitions(
    gko::array<IndexType>* partitions_in, IndexType split_index);

template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType> compute_arrow_submatrix_11(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_partitions<IndexType>& partitions);

template <typename ValueType, typename IndexType>
arrow_submatrix_12<ValueType, IndexType> compute_arrow_submatrix_12(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions);

template <typename ValueType, typename IndexType>
arrow_submatrix_21<ValueType, IndexType> compute_arrow_submatrix_21(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions);

template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType> compute_arrow_submatrix_22(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    arrow_partitions<IndexType>& partitions);

}  // namespace factorization
}  // namespace gko

#endif