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

namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {

const double PIVOT_THRESHOLD = 1e-11;
const double PIVOT_AUGMENTATION = 1e-8;  // officially it is sqrt(eps)*||A||_1
const gko::size_type MAX_DENSE_BLOCK_SIZE = 300;

template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class ArrowLu : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args&&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        std::shared_ptr<Arrow<ValueType, IndexType>>
            GKO_FACTORY_PARAMETER_SCALAR(matrix, nullptr);

        std::shared_ptr<Arrow<ValueType, IndexType>>
            GKO_FACTORY_PARAMETER_SCALAR(l_factor, nullptr);

        std::shared_ptr<Arrow<ValueType, IndexType>>
            GKO_FACTORY_PARAMETER_SCALAR(u_factor, nullptr);

        std::shared_ptr<array<IndexType>> GKO_FACTORY_PARAMETER_SCALAR(
            partitions, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ArrowLu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    ArrowLu(const Factory* factory,
            std::shared_ptr<const gko::LinOp> system_matrix,
            array<IndexType> partitions)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        generate(system_matrix, partitions, parameters_.skip_sorting,
                 parameters_.both_factors)
            ->move_to(this);
    }

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors) const;
};


}  // namespace factorization
}  // namespace gko


#endif