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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_


#include <type_traits>


#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/type_list.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <typename LSolverType, typename USolverType, bool ReverseApply,
          typename IndexType>
struct Generic<typename gko::preconditioner::Ilu<
                   LSolverType, USolverType, ReverseApply, IndexType>::Factory,
               gko::preconditioner::Ilu<LSolverType, USolverType, ReverseApply,
                                        IndexType>> {
    using type = std::shared_ptr<typename gko::preconditioner::Ilu<
        LSolverType, USolverType, ReverseApply, IndexType>::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(
                PACK(gko::preconditioner::Ilu<LSolverType, USolverType,
                                              ReverseApply, IndexType>),
                manager, item, exec, linop);
            SET_POINTER(typename LSolverType::Factory, l_solver_factory);
            SET_POINTER(typename USolverType::Factory, u_solver_factory);
            SET_POINTER(LinOpFactory, factorization_factory);
            SET_EXECUTOR;
        }();
        return std::move(ptr);
    }
};


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Ilu,
                               PACK(typename LSolverType, typename USolverType,
                                    bool ReverseApply, typename IndexType),
                               PACK(LSolverType, USolverType, ReverseApply,
                                    IndexType));


ENABLE_SELECTION_ID(ilu_factory_select, call,
                    std::shared_ptr<gko::LinOpFactory>, get_actual_factory_type,
                    RM_LinOp, Ilu);
ENABLE_SELECTION_ID(ilu_select, call, std::shared_ptr<gko::LinOp>,
                    get_actual_type, RM_LinOp, Ilu);


constexpr auto ilu_list =
    typename span_list<gko::solver::LowerTrs<>, gko::solver::UpperTrs<>,
                       tt_list<std::true_type, std::false_type>,
                       tt_list_g_t<handle_type::IndexType>>::type();


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IluFactory, gko::LinOpFactory>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        get_value_with_default(item, "LSolverType",
                               std::string("solver::LowerTrs<>")),
        get_value_with_default(item, "USolverType",
                               std::string("solver::UpperTrs<>")),
        get_value_with_default(item, "ReverseApply", std::string("false")),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    auto ptr = ilu_factory_select<type_list>(
        ilu_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Ilu, gko::LinOp>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        get_value_with_default(item, "LSolverType",
                               std::string("solver::LowerTrs<>")),
        get_value_with_default(item, "USolverType",
                               std::string("solver::UpperTrs<>")),
        get_value_with_default(item, "ReverseApply", std::string("false")),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    auto ptr = ilu_select<type_list>(
        ilu_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_
