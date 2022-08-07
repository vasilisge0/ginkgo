
// Triangular solve kernels for global matrix.
template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_1(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    using dense = matrix::Dense<ValueType>;
    const auto partition_idxs = internal_mtx.partitions_.get_data();
    const auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    const IndexType num_rhs = 1;  // 1 rhs is used
    for (auto block = 0; block < num_blocks; block++) {
        const auto block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {static_cast<size_type>(block_size),
                          static_cast<size_type>(block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(block_size),
                          static_cast<size_type>(num_rhs)};

        auto system_mtx = share(dense::create(exec));
        as<ConvertibleTo<dense>>(internal_mtx.submtx_11_.l_factors[block].get())
            ->convert_to(system_mtx.get());

        const auto values_l_factor = system_mtx->get_values();
        const auto values_rhs =
            &rhs->get_values()[partition_idxs[block] * num_rhs];
        lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                      values_rhs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_1_KERNEL);


template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_2(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    const auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    const auto partition_idxs = internal_mtx.partitions_.get_data();
    auto block_col_ptrs_idxs = internal_mtx.submtx_21_.block_ptrs.get_data();
    const auto values_l_factor =
        internal_mtx.submtx_21_.mtx.get()->get_values();
    const IndexType num_rhs = 1;
    // for (auto block = 0; block < num_blocks; block++) {
    //    auto col_idx = block_col_ptrs_idxs[block];
    //    IndexType block_size =
    //        partition_idxs[block + 1] - partition_idxs[block];
    //    dim<2> dim_tmp = {static_cast<size_type>(
    //            (block_col_ptrs_idxs[block + 1] - block_col_ptrs_idxs[block])
    //            / block_size),
    //        static_cast<size_type>(block_size)};
    //    dim<2> dim_rhs = {static_cast<size_type>(num_rhs),
    //                      static_cast<size_type>(block_size)};
    //    auto values_21 = &values_l_factor[col_idx];
    //    auto values_rhs = &rhs->get_values()[partition_idxs[block]];
    //    //csc_spdgemm<ValueType, IndexType>(dim_tmp, values_l_factor, dim_rhs,
    //    //                                  values_rhs, -1.0);
    //}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_2_KERNEL);


template <typename ValueType, typename IndexType>
void lower_triangular_solve_step_3(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    using dense = matrix::Dense<ValueType>;
    auto split_index = internal_mtx.submtx_22_.split_index;
    IndexType num_rhs = 1;
    dim<2> dim_tmp = internal_mtx.submtx_22_.size;
    dim<2> dim_rhs = {dim_tmp[0], static_cast<size_type>(num_rhs)};
    auto l_factor = share(dense::create(exec));
    as<ConvertibleTo<dense>>(internal_mtx.submtx_22_.l_factor.get())
        ->convert_to(l_factor.get());
    auto values_l_factor = l_factor.get()->get_values();
    auto values_rhs = &rhs->get_values()[split_index];
    lower_triangular_solve_kernel(dim_tmp, values_l_factor, dim_rhs,
                                  values_rhs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRIANGULAR_SOLVE_STEP_3_KERNEL);


template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_1(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    using dense = matrix::Dense<ValueType>;
    auto split_index = internal_mtx.submtx_22_.split_index;
    IndexType num_rhs = 1;
    const dim<2> dim_tmp = internal_mtx.submtx_22_.size;
    const dim<2> dim_rhs = {dim_tmp[0], static_cast<size_type>(num_rhs)};
    const auto u_factor = share(dense::create(exec));
    as<ConvertibleTo<dense>>(internal_mtx.submtx_22_.u_factor.get())
        ->convert_to(u_factor.get());

    const auto values_u_factor = u_factor.get()->get_const_values();
    auto values_rhs = &rhs->get_values()[split_index];
    lower_triangular_solve_kernel(dim_tmp, values_u_factor, dim_rhs,
                                  values_rhs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_1_KERNEL);


template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_2(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    const auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    const auto partition_idxs = internal_mtx.partitions_.get_data();
    const auto block_ptrs = internal_mtx.submtx_12_.block_ptrs.get_const_data();
    const auto split_index = internal_mtx.submtx_12_.split_index;
    size_type num_rhs = 1;
    auto values_u_factor = internal_mtx.submtx_12_.mtx.get()->get_values();
    // for (auto block = 0; block < num_blocks; block++) {
    //    const auto row_idx = block_row_ptrs_idxs[block];
    //    const auto num_rows =
    //        static_cast<size_type>(partition_idxs[block + 1] -
    //        partition_idxs[block]);
    //    const auto num_cols = static_cast<size_type>(block_ptrs[block + 1] -
    //    block_ptrs[block]) / num_rows; const dim<2> dim_tmp = {num_rows,
    //    num_colsj}; const dim<2> dim_rhs = {block_size,
    //    static_cast<size_type>(num_rhs)}; const auto values_12 =
    //    &values_u_factor[row_idx]; auto values_rhs =
    //    &rhs->get_values()[split_index];
    //    // csr_spdgemm<ValueType, IndexType>(dim_tmp, values_12, dim_rhs,
    //    //                                     values_rhs, -1.0);
    //}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_2_KERNEL);


template <typename ValueType, typename IndexType>
void upper_triangular_solve_step_3(
    std::shared_ptr<const DefaultExecutor> exec,
    factorization::arrow_matrix<ValueType, IndexType>& internal_mtx,
    matrix::Dense<ValueType>* rhs)
{
    const auto partition_idxs = internal_mtx.partitions_.get_data();
    // auto u_factors = internal_mtx.submtx_11_.u_factors;
    const auto num_blocks = internal_mtx.submtx_11_.num_blocks;
    IndexType num_rhs = 1;  // 1 rhs is used
    for (auto block = 0; block < num_blocks; block++) {
        IndexType block_size =
            partition_idxs[block + 1] - partition_idxs[block];
        dim<2> dim_tmp = {static_cast<size_type>(block_size),
                          static_cast<size_type>(block_size)};
        dim<2> dim_rhs = {static_cast<size_type>(block_size),
                          static_cast<size_type>(num_rhs)};
        // auto values_u_factor =
        //    u_factors[block]->get_values();
        // auto values_rhs = &rhs->get_values()[partition_idxs[block] *
        // num_rhs]; lower_triangular_solve_kernel(dim_tmp, values_u_factor,
        // dim_rhs,
        //                              values_rhs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRIANGULAR_SOLVE_STEP_3_KERNEL);