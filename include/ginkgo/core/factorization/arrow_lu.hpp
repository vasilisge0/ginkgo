#ifndef ARROW_LU_HPP
#define ARROW_LU_HPP

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <memory>
// #include <ginkgo/core/matrix/arrow.hpp>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
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

#include "core/factorization/arrow_matrix.hpp"

// namespace gko {
// namespace matrix {

// template <typename ValueType, typename IndexType>
// class Arrow;

// }
// }

namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {

const double PIVOT_THRESHOLD = 1e-11;
const double PIVOT_AUGMENTATION = 1e-8;  // officially it is sqrt(eps)*||A||_1

// Declaration of arrow_lu_workspace struct.


/**
 * Represents an incomplete Cholesky factorization (ArrowLu(0)) of a sparse
 * matrix.
 *
 * More specifically, it consists of a lower triangular factor $L$ and
 * its conjugate transpose $L^H$ with sparsity pattern
 * $\mathcal S(L + L^H)$ = $\mathcal S(A)$
 * fulfilling $LL^H = A$ at every non-zero location of $A$.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class ArrowLu : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const matrix_type> get_lt_factor() const
    {
        if (this->get_operators().size() == 2) {
            // Can be `static_cast` since the type is guaranteed in this class
            return std::static_pointer_cast<const matrix_type>(
                this->get_operators()[1]);
        } else {
            return std::static_pointer_cast<const matrix_type>(
                share(get_l_factor()->conj_transpose()));
        }
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args&&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * `true` will generate both L and L^H, `false` will only generate the L
         * factor, resulting in a Composition of only a single LinOp. This can
         * be used to avoid the transposition operation.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(both_factors, true);

        // arrow_lu_workspace<ValueType, IndexType> workspace;
        std::shared_ptr<arrow_lu_workspace<ValueType, IndexType>>
            GKO_FACTORY_PARAMETER_SCALAR(workspace, nullptr);

        // std::shared_ptr<matrix::Arrow<ValueType, IndexType>>
        // GKO_FACTORY_PARAMETER_SCALAR(
        //     mtx, nullptr);

        int GKO_FACTORY_PARAMETER_SCALAR(test, 1);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ArrowLu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    // , arrow_partitions<IndexType> partitions
    // ArrowLu(const Factory* factory, std::shared_ptr<const gko::LinOp>
    // system_matrix)
    //     : Composition<ValueType>{factory->get_executor()},
    //       parameters_{factory->get_parameters()}
    // {
    //     if (parameters_.l_strategy == nullptr) {
    //         parameters_.l_strategy =
    //             std::make_shared<typename matrix_type::classical>();
    //     }
    //     generate(system_matrix, parameters_.skip_sorting,
    //              parameters_.both_factors)
    //         ->move_to(this);
    // }

    ArrowLu(const Factory* factory,
            std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate_workspace(system_matrix);
        // using CsrMatrix = matrix::Csr<ValueType, IndexType>;
        // auto exec = factory->get_executor();
        // auto csr_system_matrix = share(CsrMatrix::create(exec));
        //     as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        //     ->convert_to(csr_system_matrix.get());
    }

    // void apply_impl(const LinOp* b, LinOp* x) const override;

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors) const;

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors, arrow_partitions<IndexType>& partitions,
        std::shared_ptr<arrow_lu_workspace<ValueType, IndexType>> workspace)
        const;

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors, std::ifstream& infile) const;

    void generate_workspace(const std::shared_ptr<const LinOp>& system_matrix);
};


}  // namespace factorization
}  // namespace gko


#endif