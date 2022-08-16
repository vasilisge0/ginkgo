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


namespace gko {
namespace factorization {


const gko::size_type MAX_DENSE_BLOCK_SIZE = 300;

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
    size_type num_elems;
    size_type num_blocks;
    array<IndexType> rows;
    array<IndexType> row_ptrs;
    array<IndexType> block_ptrs;
    block_csr_storage();
    block_csr_storage(std::shared_ptr<const Executor> exec,
                      IndexType num_elems_in, IndexType num_blocks_in);
    block_csr_storage(array<IndexType>& rows_in, array<IndexType>& row_ptrs_in,
                      array<IndexType>& block_ptrs_in);
    block_csr_storage(std::shared_ptr<const Executor> exec,
                      IndexType num_elems_in, IndexType num_blocks_in,
                      IndexType* rows_in, IndexType* row_ptrs_in,
                      IndexType* block_row_ptrs_in);
    void reset_block_ptrs();
    void resize();
};

template <typename IndexType>
struct arrow_partitions {
public:
    dim<2> size;
    size_type split_index = 0;
    size_type num_endpoints = 0;
    array<IndexType> data;

    arrow_partitions();
    arrow_partitions(arrow_partitions<IndexType>& in);
    arrow_partitions(std::shared_ptr<const Executor> exec,
                     std::ifstream& infile);
    arrow_partitions(gko::array<IndexType>& partition_idxs,
                     size_type split_index_in);
    arrow_partitions(gko::array<IndexType>* partition_idxs,
                     size_type split_index_in);
    arrow_partitions(std::unique_ptr<gko::array<IndexType>> partition_idxs,
                     size_type split_index_in);
    arrow_partitions(const arrow_partitions& partitions_in);
    void read(std::ifstream& infile);
    IndexType* get_data();
    const IndexType* get_const_data() const;
    IndexType get_num_blocks();
};

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

template <typename ValueType, typename IndexType>
struct arrow_submatrix {
public:
    std::shared_ptr<const Executor> exec = nullptr;
    dim<2> size = {0, 0};
    size_type nnz = size_type{};
    size_type num_blocks = size_type{};
    size_type max_block_size = size_type{};
    size_type max_block_nnz = size_type{};
    size_type split_index;
    array<IndexType> nnz_per_block;
    array<IndexType> block_ptrs;
    arrow_submatrix() {}
    arrow_submatrix(std::shared_ptr<const Executor> exec_in, dim<2> size_in,
                    size_type nnz_in, size_type num_blocks_in,
                    size_type max_block_size_in, size_type max_block_nnz_in,
                    size_type split_index_in);
    arrow_submatrix(const arrow_submatrix<ValueType, IndexType>& submtx_in);
    arrow_submatrix(const arrow_submatrix<ValueType, IndexType>&& submtx_in);
};

template <typename ValueType, typename IndexType>
struct arrow_submatrix_11 : public arrow_submatrix<ValueType, IndexType> {
    size_type nnz_l = size_type{};
    size_type nnz_u = size_type{};
    array<IndexType> row_ptrs_cur;
    std::vector<std::unique_ptr<gko::LinOp>> l_factors;
    std::vector<std::unique_ptr<gko::LinOp>> u_factors;
    std::vector<std::unique_ptr<gko::LinOp>> diag_blocks;
    arrow_submatrix_11(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                       arrow_partitions<IndexType>& arrow_partitions);
    arrow_submatrix_11(
        const arrow_submatrix_11<ValueType, IndexType>& submtx_in);
    // arrow_submatrix_11<ValueType, IndexType>&
    // operator=(arrow_submatrix_11<ValueType, IndexType> &&rhs);
    arrow_submatrix_11<ValueType, IndexType>& operator=(
        arrow_submatrix_11<ValueType, IndexType>&& rhs);
};

template <typename ValueType, typename IndexType>
struct arrow_submatrix_12 : public arrow_submatrix<ValueType, IndexType> {
    using csr_mtx = matrix::Csr<ValueType, IndexType>;
    std::shared_ptr<csr_mtx> mtx;
    array<IndexType> row_ptrs_cur;
    arrow_submatrix_12(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                       arrow_submatrix_11<ValueType, IndexType>& submtx_11,
                       arrow_partitions<IndexType>& partitions);
};

template <typename ValueType, typename IndexType>
struct arrow_submatrix_21 : public arrow_submatrix<ValueType, IndexType> {
    using csr_mtx = matrix::Csr<ValueType, IndexType>;
    std::shared_ptr<csr_mtx> mtx;
    std::shared_ptr<block_csr_storage<IndexType>> block_storage;
    array<IndexType> row_ptrs_cur;
    array<IndexType> row_ptrs_cur2;
    array<IndexType> col_ptrs_cur;
    arrow_submatrix_21(
        const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
        const arrow_submatrix_11<ValueType, IndexType>& submtx_11,
        const arrow_partitions<IndexType>& partitions);
};

template <typename ValueType, typename IndexType>
struct arrow_submatrix_22 : public arrow_submatrix<ValueType, IndexType> {
    using dense_mtx = matrix::Dense<ValueType>;
    std::shared_ptr<gko::LinOp> mtx;
    std::shared_ptr<gko::LinOp> l_factor;
    std::shared_ptr<gko::LinOp> u_factor;
    arrow_submatrix_22(
        const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
        const arrow_submatrix_11<ValueType, IndexType>& submtx_11,
        const arrow_submatrix_12<ValueType, IndexType>& submtx_12,
        const arrow_submatrix_21<ValueType, IndexType>& submtx_21,
        const arrow_partitions<IndexType>& partitions);
    arrow_submatrix_22(const arrow_submatrix_22<ValueType, IndexType>& input);
};


template <typename ValueType, typename IndexType>
struct ArrowLuState {
private:
public:
    arrow_matrix<ValueType, IndexType> mtx_;

    ArrowLuState(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 arrow_partitions<IndexType>& partitions);
    ArrowLuState(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 std::ifstream& instream);
    ArrowLuState(std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
                 gko::array<IndexType>& partitions, IndexType split_index_in);
    arrow_matrix<ValueType, IndexType>* get_matrix();
    arrow_submatrix<ValueType, IndexType>& get_submatrix(IndexType row_block,
                                                         IndexType col_block);
    arrow_submatrix_11<ValueType, IndexType>* const get_submatrix_11();
    arrow_submatrix_12<ValueType, IndexType>* const get_submatrix_12();
    arrow_submatrix_21<ValueType, IndexType>* const get_submatrix_21();
    arrow_submatrix_22<ValueType, IndexType>* const get_submatrix_22();
    arrow_partitions<IndexType>* get_partitions();
};

template <typename IndexType>
arrow_partitions<IndexType> compute_partitions(
    gko::array<IndexType>* partitions_in, IndexType split_index);

}  // namespace factorization
}  // namespace gko

#endif