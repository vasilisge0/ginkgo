


#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_11_FACTORIZE_KERNEL(_type) \
    void submatrix_11::factorize(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,  \
        Partitions<IndexType>& partitions)

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_12_FACTORIZE_KERNEL(_type) \
    void submatrix_12::factorize(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,  \
        Partitions<IndexType>& partitions)

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_21_FACTORIZE_KERNEL(_type) \
    void submatrix_21::factorize(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,  \
        Partitions<IndexType>& partitions)

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_22_FACTORIZE_KERNEL(_type) \
    void submatrix_22::factorize()

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_11_ASSEMBLE_KERNEL(_type) \
    void submatrix_11::assemble(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \ 
        Partitions<IndexType>& partitions);

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_12_ASSEMBLE_KERNEL(_type) \
    void submatrix_12::assemble(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \ 
    Partitions<IndexType>& partitions);

#define GKO_DECLARE_BLOCK_ARROW_LU_SUBMATRIX_21_ASSEMBLE_KERNEL(_type) \
    void submatrix_21::assemble(                                       \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \ 
    Partitions<IndexType>& partitions);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_12_COMPUTE_REMAINING_NZ_KERNEL( \
    _type)                                                                \
    void submatrix_12::compute_remaining_nz(                              \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,    \
        IndexType* partition_idxs);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_21_COMPUTE_REMAINING_NZ_KERNEL( \
    _type)                                                                \
    void submatrix_21::compute_remaining_nz(                              \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,    \
        IndexType* partition_idxs);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_12_FIND_MIN_COL_KERNEL(_type) \
    void submatrix_12::find_min_col(                                    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,  \
        IndexType* partition_idxs);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_21_FIND_MIN_COL_KERNEL(_type) \
    void submatrix_21::find_min_col(                                    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx,  \
        IndexType* partition_idxs);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_12_SET_ENTRIES_KERNEL(_type) \
    void submatrix_12::set_entries(                                    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \
        IndexType* partitions);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_21_SET_ENTRIES_KERNEL(_type) \
    void submatrix_21::set_entries(                                    \
        std::shared_ptr<matrix::Csr<ValueType, IndexType>> global_mtx, \
        IndexType* partitions);

#define GKO_DECLARE_BLOCK_ARROW_SUBMATRIX_22_COMPUTE_SCHUR_COMPLEMENT_KERNEL \
    void compute_schur_complement(                                           \
        submatrix_11<ValueType, IndexType>& submtx_11,                       \
        submatrix_12<ValueType, IndexType>& submtx_12,                       \
        submatrix_21<ValueType, IndexType>& submtx_21,                       \
        Partitions<IndexType>& partitions);