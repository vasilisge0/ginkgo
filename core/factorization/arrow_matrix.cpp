#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/arrow_lu.hpp>

#include "core/factorization/arrow_lu_kernels.hpp"
#include "core/factorization/arrow_matrix.hpp"


namespace gko {
namespace factorization {

// arrow_matrix struct. This struct is used for managing data used by the
// arrow_lu factorization and solve methods.

// constructor 1
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

#define GKO_DECLARE_ARROW_MATRIX_CONSTRUCTOR_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>::arrow_matrix(                     \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,           \
        std::ifstream& infile);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_MATRIX_CONSTRUCTOR_KERNEL);

// constructor 2
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

// constructor 3
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

#define GKO_DECLARE_ARROW_MATRIX_CONSTRUCTOR3_KERNEL(ValueType, IndexType) \
    arrow_matrix<ValueType, IndexType>::arrow_matrix(                      \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> mtx,            \
        gko::array<IndexType>& partition_idxs, IndexType split_index_in)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_MATRIX_CONSTRUCTOR3_KERNEL);


// arrow_submatrix_11 struct. Manages entries in (1, 1) sub-block (1-indexing)
// of global_mtx.

// constructor
template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_partitions<IndexType>& partitions)
{
    auto d_tmp = partitions.data.get_data();
    exec = global_mtx->get_executor();
    split_index = partitions.split_index;
    num_blocks = partitions.num_endpoints - 1;
    row_ptrs_tmp = array<IndexType>(exec, partitions.split_index + 1);
    std::cout << "num_blocks: " << num_blocks << '\n';
    dense_l_factors.reserve(num_blocks);
    dense_u_factors.reserve(num_blocks);
    dense_diagonal_blocks.reserve(num_blocks);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_KERNEL(ValueType,   \
                                                          IndexType)   \
    arrow_submatrix_11<ValueType, IndexType>::arrow_submatrix_11(      \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_11_CONSTRUCTOR_KERNEL);


template <typename ValueType, typename IndexType>
arrow_submatrix_11<ValueType, IndexType> compute_arrow_submatrix_11(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_partitions<IndexType>& partitions)
{
    auto submtx_11 =
        arrow_submatrix_11<ValueType, IndexType>(global_mtx, partitions);
    return submtx_11;
}

#define GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_11_KERNEL(ValueType, IndexType) \
    arrow_submatrix_11<ValueType, IndexType> compute_arrow_submatrix_11(    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,      \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_11_KERNEL)


// arrow_submatrix_12 struct. Manages entries in (1, 2) sub-block (1-indexing)
// of global_mtx.

// constructor
template <typename ValueType, typename IndexType>
arrow_submatrix_12<ValueType, IndexType>::arrow_submatrix_12(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions)
{
    // Initializes executor and dimensions related data
    exec = submtx_11.exec;
    split_index = partitions.split_index;
    num_blocks = partitions.num_endpoints - 1;
    size = {static_cast<size_type>(split_index),
            static_cast<size_type>(global_mtx->get_size()[0] - split_index)};

    // Allocates temporary data.
    row_ptrs_tmp = array<IndexType>(exec, size[0]);
    row_ptrs_tmp.fill(0);
    block_row_ptrs = array<IndexType>(exec, num_blocks + 1);
    block_row_ptrs.fill(0);
    nz_per_block = array<IndexType>(exec, num_blocks);
    nz_per_block.fill(0);
}

// arrow_submatrix_21. Manages entries in (1, 2) sub-block (1-indexing) of
// global_mtx.

// constructor
template <typename ValueType, typename IndexType>
arrow_submatrix_21<ValueType, IndexType>::arrow_submatrix_21(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions)
{
    // initializes executor and dimensions related data
    exec = submtx_11.exec;
    split_index = partitions.split_index;
    size = {global_mtx->get_size()[0] - split_index, split_index};
    num_blocks = partitions.num_endpoints - 1;

    // allocates temporary arrays
    row_ptrs_tmp = array<IndexType>(exec, size[0] + 1);
    row_ptrs_tmp.fill(0);
    row_ptrs_tmp2 = array<IndexType>(exec, size[0] + 1);
    row_ptrs_tmp2.fill(0);
    col_ptrs_tmp = array<IndexType>(exec, size[1] + 1);
    col_ptrs_tmp.fill(0);

    // allocate block_col_ptrs and nz_per_block arrays
    block_col_ptrs = array<IndexType>(exec, num_blocks + 1);
    block_col_ptrs.fill(0);
    nz_per_block = {exec, static_cast<size_type>(num_blocks)};
    nz_per_block.fill(0);
}

template <typename ValueType, typename IndexType>
arrow_submatrix_21<ValueType, IndexType> compute_arrow_submatrix_21(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions)
{
    auto submtx_21 = arrow_submatrix_21<ValueType, IndexType>(
        global_mtx, submtx_11, partitions);
    return submtx_21;
}

#define GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_21_KERNEL(ValueType, IndexType) \
    arrow_submatrix_21<ValueType, IndexType> compute_arrow_submatrix_21(    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,      \
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,                \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_21_KERNEL)

template <typename ValueType, typename IndexType>
arrow_submatrix_12<ValueType, IndexType> compute_arrow_submatrix_12(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_partitions<IndexType>& partitions)
{
    auto submtx_12 = arrow_submatrix_12<ValueType, IndexType>(
        global_mtx, submtx_11, partitions);
    return submtx_12;
}

#define GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_12_KERNEL(ValueType, IndexType) \
    arrow_submatrix_12<ValueType, IndexType> compute_arrow_submatrix_12(    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,      \
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,                \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_12_KERNEL)

// arrow_submatrix_22. Manages entries in (1, 2) sub-block (1-indexing) of
// global_mtx.

// constructor
template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    arrow_partitions<IndexType>& partitions)
{
    // initializes executor and dimensions related data
    exec = submtx_11.exec;
    split_index = partitions.split_index;
    size = {global_mtx->get_size()[0] - static_cast<size_type>(split_index),
            global_mtx->get_size()[1] - static_cast<size_type>(split_index)};

    // allocates memory for values and factors
    auto values = array<ValueType>(exec, size[0] * size[1]);
    auto values_l_t = array<ValueType>(exec, size[0] * size[1]);
    auto values_u_t = array<ValueType>(exec, size[0] * size[1]);
    values.fill(0.0);
    values_l_t.fill(0.0);
    values_u_t.fill(0.0);
    IndexType stride = 1;
    schur_complement =
        matrix::Dense<ValueType>::create(exec, size, std::move(values), stride);
    dense_u_factor = matrix::Dense<ValueType>::create(
        exec, size, std::move(values_u_t), stride);
    dense_l_factor = matrix::Dense<ValueType>::create(
        exec, size, std::move(values_l_t), stride);
}

#define GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR0_KERNEL(ValueType,  \
                                                           IndexType)  \
    arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(      \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,           \
        arrow_submatrix_12<ValueType, IndexType>& submtx_12,           \
        arrow_submatrix_21<ValueType, IndexType>& submtx_21,           \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR0_KERNEL)

// move constructor
template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(
    const arrow_submatrix_22<ValueType, IndexType>& input)
    : schur_complement(std::move(input.schur_complement)),
      dense_u_factor(std::move(input.dense_l_factor)),
      dense_l_factor(std::move(input.dense_u_factor)),
      exec(input.exec),
      split_index(input.split_index),
      size(input.size)
{}

#define GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR1_KERNEL(ValueType, \
                                                           IndexType) \
    arrow_submatrix_22<ValueType, IndexType>::arrow_submatrix_22(     \
        const arrow_submatrix_22<ValueType, IndexType>& input);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ARROW_SUBMATRIX_22_CONSTRUCTOR1_KERNEL)

template <typename ValueType, typename IndexType>
arrow_submatrix_22<ValueType, IndexType> compute_arrow_submatrix_22(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,
    arrow_submatrix_11<ValueType, IndexType>& submtx_11,
    arrow_submatrix_12<ValueType, IndexType>& submtx_12,
    arrow_submatrix_21<ValueType, IndexType>& submtx_21,
    arrow_partitions<IndexType>& partitions)
{
    auto submtx_22 = arrow_submatrix_22<ValueType, IndexType>(
        global_mtx, submtx_11, submtx_12, submtx_21, partitions);
    return submtx_22;
}

#define GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_22_KERNEL(ValueType, IndexType) \
    arrow_submatrix_22<ValueType, IndexType> compute_arrow_submatrix_22(    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,      \
        arrow_submatrix_11<ValueType, IndexType>& submtx_11,                \
        arrow_submatrix_12<ValueType, IndexType>& submtx_12,                \
        arrow_submatrix_21<ValueType, IndexType>& submtx_21,                \
        arrow_partitions<IndexType>& partitions);

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_ARROW_SUBMATRIX_22_KERNEL)


// arrow_partitions struct. A wrapper on top of gko::array.

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    const arrow_partitions<IndexType>& partitions_in)
{
    split_index = partitions_in.split_index;
    num_endpoints = partitions_in.num_endpoints;
    size = partitions_in.size;
    data = std::move(partitions_in.data);
}


template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions()
{}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR0_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions();

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR0_KERNEL);

template <typename IndexType>
arrow_partitions<IndexType> compute_partitions(
    gko::array<IndexType>* partitions_in, IndexType split_index)
{
    // auto partitions = arrow_partitions<IndexType>();
    std::cout << "in compute_partitions\n";
    arrow_partitions<IndexType> partitions(partitions_in, split_index);
    std::cout << "beore return\n";
    return partitions;
}

#define GKO_DECLARE_COMPUTE_PARTITIONS_KERNEL(IndexType) \
    arrow_partitions<IndexType> compute_partitions(      \
        gko::array<IndexType>* partitions_in, IndexType split_index)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_COMPUTE_PARTITIONS_KERNEL);

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(arrow_partitions<IndexType>& in)
{
    this->split_index = in.split_index;
    this->num_endpoints = in.num_endpoints;
    this->size = in.size;
    this->data = std::move(in.data);
}

// commented temporary
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
    auto tmp = data.get_data()[0];
    auto index = 0;
    while (tmp < split_index) {
        index += 1;
        // tmp = data.get_data()[index];
    }
    num_endpoints = index - 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                 \
        std::shared_ptr<const Executor> exec, std::ifstream& infile);

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    gko::array<IndexType>& partition_idxs, IndexType split_index_in)
{
    // exec = partition_idxs.get_executor();
    data = std::move(partition_idxs);
    split_index = split_index_in;
    size = {data.get_num_elems(), 1};
    auto tmp = data.get_data()[0];
    auto index = 0;
    while (tmp < split_index) {
        index += 1;
        tmp = data.get_data()[index];
    }
    num_endpoints = index + 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR2_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                  \
        gko::array<IndexType>& partition_idxs, IndexType split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR2_KERNEL);

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    gko::array<IndexType>* partition_idxs, IndexType split_index_in)
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

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR00_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                   \
        gko::array<IndexType>* partition_idxs, IndexType split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR00_KERNEL);

template <typename IndexType>
arrow_partitions<IndexType>::arrow_partitions(
    std::unique_ptr<gko::array<IndexType>> partition_idxs,
    IndexType split_index_in)
{
    // exec = partition_idxs.get_executor();
    data = std::move(*(partition_idxs.get()));
    split_index = split_index_in;
    size = {data.get_num_elems(), 1};
    auto tmp = data.get_data()[0];
    auto index = 0;
    while (tmp < split_index) {
        index += 1;
        tmp = data.get_data()[index];
    }
    num_endpoints = index + 1;
}

#define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR3_KERNEL(IndexType) \
    arrow_partitions<IndexType>::arrow_partitions(                  \
        std::unique_ptr<gko::array<IndexType>> partition_idxs,      \
        IndexType split_index_in)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR3_KERNEL);


// #define GKO_DECLARE_ARROW_PARTITIONS_CONSTRUCTOR2_KERNEL(IndexType) \
//     arrow_partitions<IndexType>::arrow_partitions(gko::array<IndexType> partition_idxs, \
//         dim<2> size_in, IndexType num_endpoints_in);


}  // namespace factorization
}  // namespace gko