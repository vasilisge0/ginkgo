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

#ifndef GKO_PUBLIC_CORE_MATRIX_ARROW_HPP_
#define GKO_PUBLIC_CORE_MATRIX_ARROW_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType>
class Diagonal;

template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;

template <typename ValueType, typename IndexType>
class Arrow;

template <typename ValueType, typename IndexType>
class Fbcsr;

template <typename ValueType, typename IndexType>
class ArrowBuilder;


namespace detail {}  // namespace detail


/**
 * Arrow is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format).
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column of each
 * nonzero element.
 *
 * The Arrow LinOp supports different operations:
 *
 * ```cpp
 * matrix::Arrow *A, *B, *C;      // matrices
 * matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
 * matrix::Dense *alpha, *beta; // scalars of dimension 1x1
 * matrix::Identity *I;         // identity matrix
 *
 * // Applying to Dense matrices computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 *
 * // Applying to Arrow matrices computes a SpGEMM product of two sparse
 * matrices A->apply(B, C)              // C = A*B A->apply(alpha, B, beta, C)
 * // C = alpha*A*B + beta*C
 *
 * // Applying to an Identity matrix computes a SpGEAM sparse matrix addition
 * A->apply(alpha, I, beta, B) // B = alpha*A + beta*B
 * ```
 * Both the SpGEMM and SpGEAM operation require the input matrices to be sorted
 * by column index, otherwise the algorithms will produce incorrect results.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup arrow
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Arrow : public EnableLinOp<Arrow<ValueType, IndexType>>,
              public EnableCreateMethod<Arrow<ValueType, IndexType>>  //,
//   public ConvertibleTo<Arrow<next_precision<ValueType>, IndexType>>, //,
//              public ConvertibleTo<Dense<ValueType>>,
//              public ConvertibleTo<Coo<ValueType, IndexType>>,
//              public ConvertibleTo<Ell<ValueType, IndexType>>,
//              public ConvertibleTo<Fbcsr<ValueType, IndexType>>,
//              public ConvertibleTo<Hybrid<ValueType, IndexType>>,
//              public ConvertibleTo<Sellp<ValueType, IndexType>>,
//              public ConvertibleTo<SparsityCsr<ValueType, IndexType>>,
//              public DiagonalExtractable<ValueType>,
//  public ReadableFromMatrixData<ValueType, IndexType>,
//  public WritableToMatrixData<ValueType, IndexType>
//              public Transposable,
//              public Permutable<IndexType>,
//              public EnableAbsoluteComputation<
//                  remove_complex<Arrow<ValueType, IndexType>>>,
// public ScaledIdentityAddable
{
    friend class EnableCreateMethod<Arrow>;
    friend class EnablePolymorphicObject<Arrow, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Diagonal<ValueType>;
    friend class Ell<ValueType, IndexType>;
    friend class Hybrid<ValueType, IndexType>;
    friend class Sellp<ValueType, IndexType>;
    friend class SparsityCsr<ValueType, IndexType>;
    friend class Fbcsr<ValueType, IndexType>;
    friend class ArrowBuilder<ValueType, IndexType>;
    //    friend class Arrow<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Arrow>::convert_to;
    using EnableLinOp<Arrow>::move_to;
    // using ReadableFromMatrixData<ValueType, IndexType>::read;
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Arrow<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Arrow>;

    // /**
    //  * Copy-assigns a Arrow matrix. Preserves executor, copies everything
    //  else.
    //  */
    // Arrow& operator=(const Arrow&);

    // /**
    //  * Move-assigns a Arrow matrix. Preserves executor, moves the data and
    //  * leaves the moved-from object in an empty state (0x0 LinOp with
    //  unchanged
    //  * executor and strategy, no nonzeros and valid row pointers).
    //  */
    // Arrow& operator=(Arrow&&);

    // //   Arrow(std::shared_ptr<const Executor> exec, array<index_type>&
    // //   partitions_in) {
    // //         this->partitions_ = std::move(partitions_in);
    // //     }

    // Arrow() {}

    // /**
    //  * Copy-constructs a Arrow matrix. Inherits executor, strategy and data.
    //  */
    // Arrow(const Arrow&);

    // /**
    //  * Move-constructs a Arrow matrix. Inherits executor and strategy, moves
    //  the
    //  * data and leaves the moved-from object in an empty state (0x0 LinOp
    //  with
    //  * unchanged executor and strategy, no nonzeros and valid row pointers).
    //  */
    // Arrow(Arrow&&);

    friend class Arrow<next_precision<ValueType>, IndexType>;

    // void convert_to(Arrow<next_precision<ValueType>, IndexType>* result)
    // const
    //     override;

    // void move_to(Arrow<next_precision<ValueType>, IndexType>* result)
    // override;
    //
    //    void convert_to(Dense<ValueType>* other) const override;
    //
    //    void move_to(Dense<ValueType>* other) override;
    //
    //    void convert_to(Coo<ValueType, IndexType>* result) const override;
    //
    //    void move_to(Coo<ValueType, IndexType>* result) override;
    //
    //    void convert_to(Ell<ValueType, IndexType>* result) const override;
    //
    //    void move_to(Ell<ValueType, IndexType>* result) override;
    //
    //    void convert_to(Fbcsr<ValueType, IndexType>* result) const override;
    //
    //    void move_to(Fbcsr<ValueType, IndexType>* result) override;
    //
    //    void convert_to(Hybrid<ValueType, IndexType>* result) const override;
    //
    //    void move_to(Hybrid<ValueType, IndexType>* result) override;
    //
    //    void convert_to(Sellp<ValueType, IndexType>* result) const override;
    //
    //    void move_to(Sellp<ValueType, IndexType>* result) override;
    //
    //    void convert_to(SparsityCsr<ValueType, IndexType>* result) const
    //    override;
    //
    //    void move_to(SparsityCsr<ValueType, IndexType>* result) override;
    //
    //    void convert_to(Arrow<ValueType, IndexType>* result) const override;
    //
    // void move_to(Arrow<ValueType, IndexType>* result) override;

    //    void read(const mat_data& data) override;
    //
    //    void read(const device_mat_data& data) override;
    //
    //    void read(device_mat_data&& data) override;
    //
    //    void write(mat_data& data) const override;
    //
    void apply_impl(const LinOp* b,
                    LinOp* x) const override GKO_NOT_IMPLEMENTED;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override GKO_NOT_IMPLEMENTED;

    //    std::unique_ptr<LinOp> transpose() const override;
    //
    //    std::unique_ptr<LinOp> conj_transpose() const override;
    //
    //    std::unique_ptr<LinOp> permute(
    //        const array<IndexType>* permutation_indices) const override;
    //
    //    std::unique_ptr<LinOp> inverse_permute(
    //        const array<IndexType>* inverse_permutation_indices) const
    //        override;
    //
    //    std::unique_ptr<LinOp> row_permute(
    //        const array<IndexType>* permutation_indices) const override;
    //
    //    std::unique_ptr<LinOp> column_permute(
    //        const array<IndexType>* permutation_indices) const override;
    //
    //    std::unique_ptr<LinOp> inverse_row_permute(
    //        const array<IndexType>* inverse_permutation_indices) const
    //        override;
    //
    //    std::unique_ptr<LinOp> inverse_column_permute(
    //        const array<IndexType>* inverse_permutation_indices) const
    //        override;
    //
    //    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const
    //    override;
    //
    //    std::unique_ptr<absolute_type> compute_absolute() const override;
    //
    //    void compute_absolute_inplace() override;

    const IndexType* get_const_partition_idxs() const
    {
        return partitions_.get_const_data();
    }

    size_type get_partitions_num_elems() const
    {
        return partitions_.get_num_elems();
    }

    void set_partitions(array<IndexType>& partitions_in)
    {
        partitions_ = partitions_in;
    }

    IndexType get_num_blocks() { return partitions_.get_num_elems() - 1; }

    std::shared_ptr<Csr<ValueType, IndexType>> get_submatrix_00() const
    {
        return submtx_00_;
    }

    std::shared_ptr<Csr<ValueType, IndexType>> get_submatrix_01() const
    {
        return submtx_01_;
    }

    std::shared_ptr<Csr<ValueType, IndexType>> get_submatrix_10() const
    {
        return submtx_10_;
    }

    std::shared_ptr<LinOp> get_submatrix_11() const { return submtx_11_; }

    void set_submatrix_00(std::shared_ptr<Csr<ValueType, IndexType>> submtx_00)
    {
        this->submtx_00_ = std::move(submtx_00);
    }

    void set_submatrix_01(std::shared_ptr<Csr<ValueType, IndexType>> submtx_01)
    {
        this->submtx_01_ = std::move(submtx_01);
    }

    void set_submatrix_10(std::shared_ptr<Csr<ValueType, IndexType>> submtx_10)
    {
        this->submtx_10_ = std::move(submtx_10);
    }

    void set_submatrix_11(std::shared_ptr<LinOp> submtx_11)
    {
        this->submtx_11_ = std::move(submtx_11);
    }

protected:
    Arrow(std::shared_ptr<const Executor> exec) : EnableLinOp<Arrow>(exec) {}

    Arrow(std::shared_ptr<const Executor> exec, array<IndexType>& partitions)
        : EnableLinOp<Arrow>(exec)
    {
        this->partitions_ = std::move(partitions);
        // this->submtx_00_ = std::make_shared<Csr<ValueType, IndexType>>();
    }

    // Arrow(std::shared_ptr<const Executor> exec, array<IndexType>& partitions,
    //    std::unique_ptr<Csr<ValueType, IndexType>> submtx_00,
    //    std::unique_ptr<Csr<ValueType, IndexType>> submtx_01,
    //    std::unique_ptr<Csr<ValueType, IndexType>> submtx_10,
    //    std::unique_ptr<LinOp> submtx_10)
    //    : EnableLinOp<Arrow>(exec)
    //{
    //    this->partitions_ = std::move(partitions);
    //    //this->submtx_00_ = std::move(submtx_00);
    //    //this->submtx_01_ = std::move(submtx_01);
    //    //this->submtx_10_ = std::move(submtx_10);
    //    //this->submtx_11_ = std::move(submtx_11);
    //}

private:
    using csr = matrix::Csr<ValueType, IndexType>;
    using dense = matrix::Dense<ValueType>;
    array<index_type> partitions_;
    std::shared_ptr<Csr<ValueType, IndexType>> submtx_00_;
    std::shared_ptr<Csr<ValueType, IndexType>> submtx_01_;
    std::shared_ptr<Csr<ValueType, IndexType>> submtx_10_;
    std::shared_ptr<LinOp> submtx_11_;

    // void add_scaled_identity_impl(const LinOp* a, const LinOp* b) override;
};

namespace detail {}  // namespace detail
}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_ARROW_HPP_
