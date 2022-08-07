#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>

#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"


namespace gko {
namespace factorization {

template <typename IndexType>
block_csr_storage<IndexType>::block_csr_storage(
    std::shared_ptr<const Executor> exec, IndexType num_elems_in,
    IndexType num_blocks_in)
{
    num_elems = num_elems_in;
    num_blocks_in = num_blocks_in;
    array<IndexType> rows = {exec, num_elems};
    array<IndexType> row_ptrs = {exec, num_elems};
    array<IndexType> block_ptrs = {exec, num_blocks};
}

#define GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_0_KERNEL(IndexType) \
    block_csr_storage<IndexType>::block_csr_storage(                  \
        std::shared_ptr<const Executor> exec, IndexType num_elems_in, \
        IndexType num_blocks_in);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_0_KERNEL);


template <typename IndexType>
block_csr_storage<IndexType>::block_csr_storage(array<IndexType>& rows_in,
                                                array<IndexType>& row_ptrs_in,
                                                array<IndexType>& block_ptrs_in)
{
    num_elems = rows_in.get_num_elems();
    num_blocks = block_ptrs.get_num_elems();
    rows = std::move(rows_in);
    row_ptrs = std::move(row_ptrs_in);
    block_ptrs = std::move(block_ptrs_in);
}

#define GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_1_KERNEL(IndexType) \
    block_csr_storage<IndexType>::block_csr_storage(                  \
        array<IndexType>& rows_in, array<IndexType>& row_ptrs_in,     \
        array<IndexType>& block_ptrs_in);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_1_KERNEL);


template <typename IndexType>
block_csr_storage<IndexType>::block_csr_storage(
    std::shared_ptr<const Executor> exec, IndexType num_elems_in,
    IndexType num_blocks_in, IndexType* rows_in, IndexType* row_ptrs_in,
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

#define GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_2_KERNEL(IndexType)        \
    block_csr_storage<IndexType>::block_csr_storage(                         \
        std::shared_ptr<const Executor> exec, IndexType num_elems_in,        \
        IndexType num_blocks_in, IndexType* rows_in, IndexType* row_ptrs_in, \
        IndexType* block_row_ptrs_in);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BLOCK_CSR_STORAGE_CONSTRUCTOR_2_KERNEL);


template <typename ValueType, typename IndexType>
arrow_matrix<ValueType, IndexType>::arrow_matrix(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    std::ifstream& infile)
    : partitions_(mtx->get_executor(), infile),
      submtx_11_(mtx, partitions_),
      submtx_12_(mtx, submtx_11_, partitions_),
      submtx_21_(mtx, submtx_11_, partitions_),
      submtx_22_(mtx, submtx_11_, submtx_12_, submtx_21_, partitions_)
{}

#define GKO_DECLARE_ARROW_MATRIX_0_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>::arrow_matrix(           \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx, \
        std::ifstream& infile);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_MATRIX_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_matrix<ValueType, IndexType>::arrow_matrix(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    arrow_partitions<IndexType>& partitions)
    : partitions_(partitions),
      submtx_11_(mtx, partitions_),
      submtx_12_(mtx, submtx_11_, partitions_),
      submtx_21_(mtx, submtx_11_, partitions_),
      submtx_22_(mtx, submtx_11_, submtx_12_, submtx_21_, partitions_)
{}

#define GKO_DECLARE_ARROW_MATRIX_1_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>::arrow_matrix(           \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx, \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_MATRIX_1_KERNEL);


template <typename ValueType, typename IndexType>
arrow_matrix<ValueType, IndexType>::arrow_matrix(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    gko::array<IndexType>& partition_idxs, IndexType split_index_in)
    : partitions_(partition_idxs, split_index_in),
      submtx_11_(mtx, partitions_),
      submtx_12_(mtx, submtx_11_, partitions_),
      submtx_21_(mtx, submtx_11_, partitions_),
      submtx_22_(mtx, submtx_11_, submtx_12_, submtx_21_, partitions_)
{}

#define GKO_DECLARE_ARROW_MATRIX_2_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>::arrow_matrix(           \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx, \
        gko::array<IndexType>& partition_idxs, IndexType split_index_in);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_MATRIX_2_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix<ValueType, IndexType>::arrow_submatrix(
    std::shared_ptr<const Executor> exec_in, dim<2> size_in, size_type nnz_in,
    size_type num_blocks_in, size_type max_block_size_in,
    size_type max_block_nnz_in, size_type split_index_in)
{
    // exec = exec_in;
    nnz = nnz_in;
    num_blocks = num_blocks_in;
    max_block_size = max_block_size_in;
    max_block_nnz = max_block_nnz_in;
}

#define GKO_DECLARE_ARROW_SUBMATRIX_0_KERNEL(ValueType, IndexType) \
    arrow_submatrix<ValueType, IndexType>::arrow_submatrix(        \
        std::shared_ptr<const Executor> exec_in, dim<2> size_in,   \
        size_type nnz_in, size_type num_blocks_in,                 \
        size_type max_block_size_in, size_type max_block_nnz_in,   \
        size_type split_index_in);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix<ValueType, IndexType>::arrow_submatrix(
    const arrow_submatrix<ValueType, IndexType>& submtx_in)
{
    exec = submtx_in.exec;
    size = submtx_in.size;
    nnz = nnz;
    num_blocks = num_blocks;
    max_block_size = max_block_size;
    max_block_nnz = max_block_nnz;
    nnz_per_block = {exec, submtx_in.nnz_per_block.get_num_elems()};
    block_ptrs = {exec, submtx_in.block_ptrs.get_num_elems()};
    exec->copy(submtx_in.nnz_per_block.get_num_elems(),
               submtx_in.nnz_per_block.get_const_data(),
               this->nnz_per_block.get_data());
    exec->copy(submtx_in.block_ptrs.get_num_elems(),
               submtx_in.block_ptrs.get_const_data(), block_ptrs.get_data());
}

#define GKO_DECLARE_ARROW_SUBMATRIX_1_KERNEL(ValueType, IndexType) \
    arrow_submatrix<ValueType, IndexType>::arrow_submatrix(        \
        const arrow_submatrix<ValueType, IndexType>& submtx_in);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_1_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix<ValueType, IndexType>::arrow_submatrix(
    const arrow_submatrix<ValueType, IndexType>&& submtx_in)
{
    exec = submtx_in.exec;
    size = submtx_in.size;
    nnz = nnz;
    num_blocks = num_blocks;
    max_block_size = max_block_size;
    max_block_nnz = max_block_nnz;
    nnz_per_block = std::move(submtx_in.nnz_per_block);
    block_ptrs = std::move(submtx_in.block_ptrs);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_2_KERNEL(ValueType, IndexType) \
    arrow_submatrix<ValueType, IndexType>::arrow_submatrix(        \
        const arrow_submatrix<ValueType, IndexType>&& submtx_in);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_2_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    arrow_partitions<IndexType>& partitions)
{
    auto d_tmp = partitions.data.get_data();
    this->exec = mtx->get_executor();
    this->split_index = partitions.split_index;
    this->num_blocks = partitions.num_endpoints - 1;
    this->row_ptrs_cur =
        array<IndexType>(this->exec, partitions.split_index + 1);
    l_factors.reserve(this->num_blocks);
    u_factors.reserve(this->num_blocks);
    diag_blocks.reserve(this->num_blocks);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_0_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(      \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,        \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(
    const arrow_submatrix_11<ValueType, IndexType>& submtx_in)
{
    this->size = submtx_in.size;
    this->nnz = submtx_in.nnz;
    this->num_blocks = submtx_in.num_blocks;
    this->max_block_size = submtx_in.max_block_size;
    this->max_block_nnz = submtx_in.max_block_nnz;
    this->nnz_l = submtx_in.nnz_l;
    this->nnz_u = submtx_in.nnz_u;
    this->split_index = submtx_in.split_index;
    this->exec = std::move(submtx_in.exec);
    this->row_ptrs_cur = std::move(submtx_in.row_ptrs_cur);
    // this->l_factors = std::move(submtx_in.l_factors);
    // u_factors = std::move(submtx_in.u_factors);
    // diag_blocks = std::move(submtx_in.diag_blocks);
    // sparse_diagonal_blocks = std::move(submtx_tmp.sparse_diagonal_blocks);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_1_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(      \
        const arrow_submatrix_11<ValueType, IndexType>& submtx_in);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_1_KERNEL);


// template <typename ValueType, typename IndexType>
// arrow_submatrix_11<ValueType, IndexType>& arrow_submatrix_11<ValueType,
// IndexType>::operator=(arrow_submatrix_11<ValueType, IndexType> &&rhs) {
//    //arrow_submatrix_11<ValueType, IndexType> tmp; // moves the array
//    //this->size = rhs.size;
//    //this->nnz = rhs.nnz;
//    //this->num_blocks = rhs.num_blocks;
//    //this->max_block_size = rhs.max_block_size;
//    //this->max_block_nnz = rhs.max_block_nnz;
//    //this->nnz_l = rhs.nnz_l;
//    //this->nnz_u = rhs.nnz_u;
//    //this->split_index = rhs.split_index;
//    //this->exec = std::move(rhs.submtx_exec);
//    //this->row_ptrs_cur = std::move(rhs.row_ptrs_cur);
//    //l_factors = std::move(rhs.l_factors);
//    //u_factors = std::move(rhs.u_factors);
//    //diag_blocks = std::move(rhs.diag_blocks);
//    return this;
//}
//
//#define GKO_DECLARE_ARROW_SUBMATRIX_11_EQUAL_OPERATOR_KERNEL(ValueType,   \
//                                                          IndexType)      \
//    arrow_submatrix_11<ValueType, IndexType>& arrow_submatrix_11<ValueType,
//    IndexType>::operator=(arrow_submatrix_11<ValueType, IndexType> &&rhs);
//
// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARROW_SUBMATRIX_11_EQUAL_OPERATOR_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_12<ValueType, IndexType>::arrow_submatrix_12(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions)
{
    this->exec = submtx_11.exec;
    this->split_index = partitions.split_index;
    this->num_blocks = partitions.num_endpoints - 1;
    this->size = {this->split_index, mtx->get_size()[0] - this->split_index};
    this->row_ptrs_cur = array<IndexType>(this->exec, this->size[0] + 1);
    this->row_ptrs_cur.fill(0);
    this->block_ptrs = array<IndexType>(this->exec, this->num_blocks + 1);
    this->block_ptrs.fill(0);
    this->nnz_per_block = array<IndexType>(this->exec, this->num_blocks);
    this->nnz_per_block.fill(0);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_12_CONSTRUCTOR_0_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_12<ValueType, IndexType>::arrow_submatrix_12(      \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,        \
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,           \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_12_CONSTRUCTOR_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_21<ValueType, IndexType>::arrow_submatrix_21(
    const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    const arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    const arrow_partitions<IndexType>& partitions)
{
    this->exec = submtx_11.exec;
    this->split_index = partitions.split_index;
    this->size = {mtx->get_size()[0] - this->split_index, this->split_index};
    this->num_blocks = partitions.num_endpoints - 1;

    this->row_ptrs_cur = array<IndexType>(this->exec, this->size[0] + 1);
    this->row_ptrs_cur.fill(0);
    this->row_ptrs_cur2 = array<IndexType>(this->exec, this->size[0] + 1);
    this->row_ptrs_cur2.fill(0);
    this->col_ptrs_cur = array<IndexType>(this->exec, this->size[1] + 1);
    this->col_ptrs_cur.fill(0);

    this->block_ptrs = array<IndexType>(this->exec, this->num_blocks + 1);
    this->block_ptrs.fill(0);
    this->nnz_per_block = {this->exec, this->num_blocks};
    this->nnz_per_block.fill(0);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_21_CONSTRUCTOR_0_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_21<ValueType, IndexType>::arrow_submatrix_21(      \
        const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,  \
        const arrow_submatrix_11<ValueType, IndexType>& submtx_11,     \
        const arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_21_CONSTRUCTOR_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(
    const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    const arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    const arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    const arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    const arrow_partitions<IndexType>& partitions)
{
    this->exec = submtx_11.exec;
    this->split_index = partitions.split_index;
    this->size = {mtx->get_size()[0] - this->split_index,
                  mtx->get_size()[1] - this->split_index};
    auto values = array<ValueType>(this->exec, this->size[0] * this->size[1]);
    auto values_l_t =
        array<ValueType>(this->exec, this->size[0] * this->size[1]);
    auto values_u_t =
        array<ValueType>(this->exec, this->size[0] * this->size[1]);
    values.fill(0.0);
    values_l_t.fill(0.0);
    values_u_t.fill(0.0);
    IndexType stride = 1;
    this->mtx = matrix::Dense<ValueType>::create(this->exec, this->size,
                                                 std::move(values), stride);
    u_factor = matrix::Dense<ValueType>::create(this->exec, this->size,
                                                std::move(values_u_t), stride);
    l_factor = matrix::Dense<ValueType>::create(this->exec, this->size,
                                                std::move(values_l_t), stride);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR_0_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(      \
        const std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,  \
        const arrow_submatrix_11<ValueType, IndexType>& submtx_11,     \
        const arrow_submatrix_12<ValueType, IndexType>& submtx_12,     \
        const arrow_submatrix_21<ValueType, IndexType>& submtx_21,     \
        const arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR_0_KERNEL)


template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(
    const arrow_submatrix_22<ValueType, IndexType>& input)
//: arrow_submatrix<ValueType, IndexType>(input.exec, input.size, input.nnz,
// input.num_blocks,
//                  input.max_block_size, input.max_block_nnz,
//                  input.split_index, input.size),
//  mtx(std::move(input.mtx)),
//  u_factor(std::move(input.u_factor)),
//  l_factor(std::move(input.l_factor))
{}

#define GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR_1_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(      \
        const arrow_submatrix_22<ValueType, IndexType>& input);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR_1_KERNEL);


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    const arrow_partitions<IndexType>& partitions_in)
{
    split_index = partitions_in.split_index;
    num_endpoints = partitions_in.num_endpoints;
    size = partitions_in.size;
    data = std::move(partitions_in.data);
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_0_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        const arrow_partitions<IndexType>& partitions_in);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_0_KERNEL);


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions()
{}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_1_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions();

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_1_KERNEL);


template <typename IndexType>
void arrow_partitions<IndexType>::read(std::ifstream& infile)
{
    // infile >> split_index;
    for (auto i = 0; i < size[0]; ++i) {
        // infile >> data.get_data()[i];
    }
}

#define GKO_DECLARE_ARROW_PARTITIONS_READ_KERNEL(IndexType) \
    void arrow_partitions<IndexType>::read(std::ifstream& infile);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ARROW_PARTITIONS_READ_KERNEL);


template <typename IndexType>
IndexType* arrow_partitions<IndexType>::get_data()
{
    return this->data.get_data();
}

#define GKO_DECLARE_ARROW_PARTITIONS_GET_DATA_KERNEL(IndexType) \
    IndexType* arrow_partitions<IndexType>::get_data();

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_GET_DATA_KERNEL);


template <typename IndexType>
const IndexType* arrow_partitions<IndexType>::get_const_data() const
{
    return this->data.get_const_data();
}

#define GKO_DECLARE_ARROW_PARTITIONS_GET_CONST_DATA_KERNEL(IndexType) \
    const IndexType* arrow_partitions<IndexType>::get_const_data() const;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_GET_CONST_DATA_KERNEL);


template <typename IndexType>
IndexType arrow_partitions<IndexType>::get_num_blocks()
{
    return this->num_endpoints - 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_GET_NUM_BLOCKS_KERNEL(IndexType) \
    IndexType arrow_partitions<IndexType>::get_num_blocks();

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_GET_NUM_BLOCKS_KERNEL);


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(arrow_partitions<IndexType>& in)
{
    this->split_index = in.split_index;
    this->num_endpoints = in.num_endpoints;
    this->size = in.size;
    this->data = std::move(in.data);
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_2_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        arrow_partitions<IndexType>& in);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_2_KERNEL)


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    std::shared_ptr<const Executor> exec, std::ifstream& infile)
{
    std::string str;
    // std::ifstream instream("test.mtx");
    // std::ifstream instream;

    // std::getline(infile, str);
    int64 num_rows;
    // infile >> num_rows;
    int64 num_cols;
    // infile >> num_cols;
    // num_endpoints = num_rows - 1;
    size[0] = num_rows - 1;
    size[1] = 1;
    data = array<IndexType>(exec, num_rows - 1);
    read(infile);
    num_endpoints = 0;
    // find num_endpoints
    auto tmp = static_cast<size_type>(data.get_data()[0]);
    size_type index = 0;
    while (tmp < split_index) {
        index += 1;
        // tmp = static_cast<size_type>(data.get_data()[index]);
    }
    num_endpoints = index - 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_3_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        std::shared_ptr<const Executor> exec, std::ifstream& infile);

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_3_KERNEL);


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    gko::array<IndexType>& partition_idxs, size_type split_index_in)
{
    // exec = partition_idxs.get_executor();
    data = std::move(partition_idxs);
    split_index = split_index_in;
    size = {data.get_num_elems(), 1};
    size_type tmp = static_cast<size_type>(data.get_data()[0]);
    size_type index = 0;
    while (tmp < split_index) {
        index += 1;
        tmp = static_cast<size_type>(data.get_data()[index]);
    }
    num_endpoints = index + 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_4_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        gko::array<IndexType>& partition_idxs, size_type split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_4_KERNEL);


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    gko::array<IndexType>* partition_idxs, size_type split_index_in)
{
    // exec = partition_idxs.get_executor();
    data = std::move(*partition_idxs);
    split_index = split_index_in;
    size = {data.get_num_elems(), 1};
    auto tmp = data.get_data()[0];
    auto index = 0;
    std::cout << " (init) tmp: " << tmp << '\n';
    std::cout << "split_index: " << split_index << '\n';
    while (tmp < split_index) {
        index += 1;
        tmp = data.get_data()[index];
    }
    num_endpoints = index + 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_5_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        gko::array<IndexType>* partition_idxs, size_type split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_5_KERNEL);

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    std::unique_ptr<gko::array<IndexType>> partition_idxs,
    size_type split_index_in)
{
    // exec = partition_idxs.get_executor();
    data = std::move(*(partition_idxs.get()));
    split_index = split_index_in;
    size = {data.get_num_elems(), 1};
    auto tmp = static_cast<size_type>(data.get_data()[0]);
    size_type index = 0;
    while (tmp < split_index) {
        index += 1;
        tmp = static_cast<size_type>(data.get_data()[index]);
    }
    num_endpoints = index + 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_6_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        std::unique_ptr<gko::array<IndexType>> partition_idxs,       \
        size_type split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_6_KERNEL);


template <typename ValueType, typename IndexType>
arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    arrow_partitions<IndexType>& partitions)
    : mtx_(mtx, partitions)
{}

#define GKO_DECLARE_ARROW_WORKSPACE_0_KERNEL(ValueType, IndexType) \
    arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(  \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,    \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_0_KERNEL);


template <typename ValueType, typename IndexType>
arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    std::ifstream& instream)
    : mtx_(mtx, instream)
{}

#define GKO_DECLARE_ARROW_WORKSPACE_1_KERNEL(ValueType, IndexType) \
    arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(  \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,    \
        std::ifstream& instream);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_1_KERNEL);


template <typename ValueType, typename IndexType>
arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,
    gko::array<IndexType>& partitions, IndexType split_index_in)
    : mtx_(mtx, partitions, split_index_in)
{}

#define GKO_DECLARE_ARROW_WORKSPACE_2_KERNEL(ValueType, IndexType) \
    arrow_lu_workspace<ValueType, IndexType>::arrow_lu_workspace(  \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,    \
        gko::array<IndexType>& partitions, IndexType split_index_in)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_2_KERNEL);


template <typename ValueType, typename IndexType>
arrow_matrix<ValueType, IndexType>*
arrow_lu_workspace<ValueType, IndexType>::get_matrix()
{
    return &mtx_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_MATRIX_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>*                                     \
    arrow_lu_workspace<ValueType, IndexType>::get_matrix()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType>* const
arrow_lu_workspace<ValueType, IndexType>::get_submatrix_11()
{
    return &mtx_.submtx_11_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_11_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_11<ValueType, IndexType>* const                    \
    arrow_lu_workspace<ValueType, IndexType>::get_submatrix_11()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_11_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_12<ValueType, IndexType>* const
arrow_lu_workspace<ValueType, IndexType>::get_submatrix_12()
{
    return &mtx_.submtx_12_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_12_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_12<ValueType, IndexType>* const                    \
    arrow_lu_workspace<ValueType, IndexType>::get_submatrix_12()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_12_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_21<ValueType, IndexType>* const
arrow_lu_workspace<ValueType, IndexType>::get_submatrix_21()
{
    return &mtx_.submtx_21_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_21_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_21<ValueType, IndexType>* const                    \
    arrow_lu_workspace<ValueType, IndexType>::get_submatrix_21()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_21_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType>* const
arrow_lu_workspace<ValueType, IndexType>::get_submatrix_22()
{
    return &mtx_.submtx_22_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_22_KERNEL(ValueType, \
                                                            IndexType) \
    arrow_submatrix_22<ValueType, IndexType>* const                    \
    arrow_lu_workspace<ValueType, IndexType>::get_submatrix_22()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_SUBMATRIX_22_KERNEL);


template <typename ValueType, typename IndexType>
arrow_partitions<IndexType>*
arrow_lu_workspace<ValueType, IndexType>::get_partitions()
{
    return &this->get_matrix()->partitions_;
}

#define GKO_DECLARE_ARROW_WORKSPACE_GET_PARTITIONS_KERNEL(ValueType, \
                                                          IndexType) \
    arrow_partitions<IndexType>*                                     \
    arrow_lu_workspace<ValueType, IndexType>::get_partitions()

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_WORKSPACE_GET_PARTITIONS_KERNEL);


}  // namespace factorization
}  // namespace gko