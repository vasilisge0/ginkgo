/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/gmres.hpp>


#include <ginkgo/core/matrix/coo.hpp>
#include <iostream>
#include <string>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace gmres {


GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2);
GKO_REGISTER_OPERATION(step_1, gmres::step_1);
GKO_REGISTER_OPERATION(step_2, gmres::step_2);


}  // namespace gmres


namespace {


template <typename ValueType>
void print_matrix(std::string name, const matrix::Dense<ValueType> *d_mtx)
{
    auto mtx =
        matrix::Dense<ValueType>::create(d_mtx->get_executor()->get_master());
    mtx->copy_from(d_mtx);

    const auto stride = mtx->get_stride();
    const auto dim = mtx->get_size();

    const auto dim0 = dim[0];
    const auto dim1 = dim[1];
    std::cout << name << "  dim = " << dim[0] << " x " << dim[1]
              << ", st = " << stride << "  ";
    std::cout << (d_mtx->get_executor() == d_mtx->get_executor()->get_master()
                      ? "ref"
                      : "cuda")
              << std::endl;
    for (auto i = 0; i < 20; ++i) {
        std::cout << '-';
    }
    std::cout << std::endl;

    for (size_type i = 0; i < dim[0]; ++i) {
        for (size_type j = 0; j < stride; ++j) {
            if (j == dim[1]) {
                std::cout << "| ";
            }
            std::cout << mtx->get_const_values()[i * stride + j] << ' ';
        }
        std::cout << std::endl;
    }

    for (auto i = 0; i < 20; ++i) {
        std::cout << '-';
    }
    std::cout << std::endl;
}


template <typename ValueType>
bool are_same_mtx(const gko::matrix::Dense<ValueType> *d_mtx1,
                  const gko::matrix::Dense<ValueType> *d_mtx2,
                  double error = 1e-12)
{
    auto mtx1 = gko::matrix::Dense<ValueType>::create(
        d_mtx1->get_executor()->get_master());
    mtx1->copy_from(d_mtx1);
    auto mtx2 = gko::matrix::Dense<ValueType>::create(
        d_mtx2->get_executor()->get_master());
    mtx2->copy_from(d_mtx2);
    auto get_error = [](const ValueType &v1, const ValueType &v2) {
        return std::abs((v1 - v2) / std::max(v1, v2));
    };
    auto size = mtx1->get_size();
    if (size != mtx2->get_size()) {
        std::cerr << "Mismatching sizes!!!\n";
        return false;
    }
    for (int j = 0; j < size[1]; ++j) {
        for (int i = 0; i < size[0]; ++i) {
            if (get_error(mtx1->at(i, j), mtx2->at(i, j)) > error) {
                std::cerr << "Problem at component (" << i << "," << j
                          << "): " << mtx1->at(i, j) << " != " << mtx2->at(i, j)
                          << " !!!\n";
                return false;
            }
            // std::cout << "All good for (" << i << "," << j << "): " <<
            // x->at(i,j) << " == " << x_host->at(i,j) << "\n";
        }
    }
    return true;
}


template <typename ValueType>
bool compare_mtx(std::string name, const gko::matrix::Dense<ValueType> *d_mtx1,
                 const gko::matrix::Dense<ValueType> *d_mtx2,
                 double error = 1e-14)
{}


template <>
bool compare_mtx(std::string name, const gko::matrix::Dense<double> *d_mtx1,
                 const gko::matrix::Dense<double> *d_mtx2, double error)
{
    if (!are_same_mtx(d_mtx1, d_mtx2, error)) {
        print_matrix(name, d_mtx1);
        print_matrix(name, d_mtx2);
    }
}


template <typename ValueType>
void apply_preconditioner(const LinOp *preconditioner,
                          matrix::Dense<ValueType> *krylov_bases,
                          matrix::Dense<ValueType> *preconditioned_vector,
                          const size_type iter)
{
    auto target_basis = krylov_bases->create_submatrix(
        span{0, krylov_bases->get_size()[0]},
        span{iter * preconditioned_vector->get_size()[1],
             (iter + 1) * preconditioned_vector->get_size()[1]});

    // Apply preconditioner
    // print_matrix("Pre-Preconditioner " + std::to_string(iter),
    // target_basis.get());
    preconditioner->apply(target_basis.get(), preconditioned_vector);
    // print_matrix("POST-Preconditioner " + std::to_string(iter),
    // preconditioned_vector);
}

}  // namespace


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    auto krylov_bases = Vector::create(
        exec, dim<2>{system_matrix_->get_size()[1],
                     (krylov_dim_ + 1) * dense_b->get_size()[1]});
    auto next_krylov_basis = Vector::create_with_config_of(dense_b);
    auto preconditioned_vector = Vector::create_with_config_of(dense_b);
    auto hessenberg = Vector::create(
        exec, dim<2>{krylov_dim_ + 1, krylov_dim_ * dense_b->get_size()[1]});
    auto givens_sin =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto givens_cos =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto residual_norm_collection =
        Vector::create(exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
    auto residual_norm =
        Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto b_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    Array<size_type> final_iter_nums(this->get_executor(),
                                     dense_b->get_size()[1]);
    auto y = Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);

    // Initialization
    exec->run(gmres::make_initialize_1(dense_b, b_norm.get(), residual.get(),
                                       givens_sin.get(), givens_cos.get(),
                                       &stop_status, krylov_dim_));
    // b_norm = norm(b)
    // residual = dense_b
    // givens_sin = givens_cos = 0
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());
    // residual = residual - Ax

    exec->run(gmres::make_initialize_2(
        residual.get(), residual_norm.get(), residual_norm_collection.get(),
        krylov_bases.get(), &final_iter_nums, krylov_dim_));
    // residual_norm = norm(residual)
    // residual_norm_collection = {residual_norm, 0, ..., 0}
    // krylov_bases(:, 1) = residual / residual_norm
    // final_iter_nums = {0, ..., 0}

    // print_matrix(std::string("dense_x init"), dense_x);
    // print_matrix(std::string("krylov p init"), krylov_bases.get());

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, residual.get());

    int total_iter = -1;
    size_type restart_iter = 0;

    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x, residual_norm.get());
        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual_norm(residual_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        if (restart_iter == krylov_dim_) {
            std::cout << "R ";
            // Restart
            exec->run(gmres::make_step_2(residual_norm_collection.get(),
                                         krylov_bases.get(), hessenberg.get(),
                                         y.get(), dense_x, &final_iter_nums,
                                         preconditioner_.get()));
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection

            // Solve x
            // x = x + preconditioner_ * krylov_bases * y
            residual->copy_from(dense_b);
            // residual = dense_b
            system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                                  residual.get());
            // residual = residual - Ax
            exec->run(gmres::make_initialize_2(
                residual.get(), residual_norm.get(),
                residual_norm_collection.get(), krylov_bases.get(),
                &final_iter_nums, krylov_dim_));
            // residual_norm = norm(residual)
            // residual_norm_collection = {residual_norm, 0, ..., 0}
            // krylov_bases(:, 1) = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            restart_iter = 0;
        }

        apply_preconditioner(preconditioner_.get(), krylov_bases.get(),
                             preconditioned_vector.get(), restart_iter);
        // preconditioned_vector = preconditioner_ *
        //                         krylov_bases(:, restart_iter)

        // Do Arnoldi and givens rotation
        auto hessenberg_iter = hessenberg->create_submatrix(
            span{0, restart_iter + 2},
            span{dense_b->get_size()[1] * restart_iter,
                 dense_b->get_size()[1] * (restart_iter + 1)});

        // Start of arnoldi
        if (false && exec->get_master() != exec) {
            auto host_ex = exec->get_master();
            const auto h_system_matrix =
                matrix::Coo<ValueType>::create(host_ex);
            h_system_matrix->copy_from(system_matrix_.get());
            const auto h_pv = Vector::create(host_ex);
            h_pv->copy_from(preconditioned_vector.get());
            auto h_nkb = Vector::create(host_ex, next_krylov_basis->get_size(),
                                        next_krylov_basis->get_stride());
            h_system_matrix->apply(h_pv.get(), h_nkb.get());

            next_krylov_basis->copy_from(h_nkb.get());
        } else {
            system_matrix_->apply(preconditioned_vector.get(),
                                  next_krylov_basis.get());
        }
        // next_krylov_basis = A * preconditioned_vector

        // print_matrix(
        //    std::string("pre_hessenberg ") + std::to_string(total_iter),
        //    hessenberg_iter.get());
        // print_matrix(std::string("pre_next ") + std::to_string(total_iter),
        //             next_krylov_basis.get());
        // print_matrix(std::string("pre_krylov ") + std::to_string(total_iter),
        //             krylov_bases.get());
        if (false && exec != exec->get_master()) {
            auto host_ex = exec->get_master();

            auto h_nkb = Vector::create(host_ex);
            h_nkb->copy_from(next_krylov_basis.get());
            auto h_gs = Vector::create(host_ex);
            h_gs->copy_from(givens_sin.get());
            auto h_gc = Vector::create(host_ex);
            h_gc->copy_from(givens_cos.get());
            auto h_rn = Vector::create(host_ex);
            h_rn->copy_from(residual_norm.get());
            auto h_rnc = Vector::create(host_ex);
            h_rnc->copy_from(residual_norm_collection.get());
            auto h_kb = Vector::create(host_ex);
            h_kb->copy_from(krylov_bases.get());
            auto h_hi = Vector::create(host_ex);
            h_hi->copy_from(hessenberg_iter.get());
            auto h_bn = Vector::create(host_ex);
            h_bn->copy_from(b_norm.get());

            stop_status.set_executor(host_ex);
            Array<size_type> h_fin(host_ex, final_iter_nums.get_num_elems());
            exec->get_master()->copy_from(
                exec.get(), final_iter_nums.get_num_elems(),
                final_iter_nums.get_const_data(), h_fin.get_data());

            host_ex->run(gmres::make_step_1(
                h_nkb.get(), h_gs.get(), h_gc.get(), h_rn.get(), h_rnc.get(),
                h_kb.get(), h_hi.get(), h_bn.get(), restart_iter, &h_fin,
                &stop_status));

            stop_status.set_executor(exec);

            exec->run(gmres::make_step_1(
                next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                residual_norm.get(), residual_norm_collection.get(),
                krylov_bases.get(), hessenberg_iter.get(), b_norm.get(),
                restart_iter, &final_iter_nums, &stop_status));


            compare_mtx("next_krylov " + std::to_string(total_iter),
                        next_krylov_basis.get(), h_nkb.get());
            compare_mtx("givens_sin " + std::to_string(total_iter),
                        givens_sin.get(), h_gs.get());
            compare_mtx("givens_cos " + std::to_string(total_iter),
                        givens_cos.get(), h_gc.get());
            compare_mtx("residual_norm " + std::to_string(total_iter),
                        residual_norm.get(), h_rn.get());
            compare_mtx(
                "residual_norm_collection " + std::to_string(total_iter),
                residual_norm_collection.get(), h_rnc.get());
            compare_mtx("krylov_bases " + std::to_string(total_iter),
                        krylov_bases.get(), h_kb.get());
            compare_mtx("hessenberg_iter " + std::to_string(total_iter),
                        hessenberg_iter.get(), h_hi.get());
            compare_mtx("b_norm " + std::to_string(total_iter), b_norm.get(),
                        h_bn.get());


            next_krylov_basis->copy_from(h_nkb.get());
            givens_sin->copy_from(h_gs.get());
            givens_cos->copy_from(h_gc.get());
            residual_norm->copy_from(h_rn.get());
            residual_norm_collection->copy_from(h_rnc.get());
            krylov_bases->copy_from(h_kb.get());
            hessenberg_iter->copy_from(h_hi.get());
            b_norm->copy_from(h_bn.get());


        } else {
            exec->run(gmres::make_step_1(
                next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                residual_norm.get(), residual_norm_collection.get(),
                krylov_bases.get(), hessenberg_iter.get(), b_norm.get(),
                restart_iter, &final_iter_nums, &stop_status));
        }
        /*
        print_matrix(
            std::string("post_hessenberg ") + std::to_string(total_iter),
            hessenberg_iter.get());
        print_matrix(std::string("post_next ") + std::to_string(total_iter),
                     next_krylov_basis.get());
        print_matrix(std::string("post_krylov ") + std::to_string(total_iter),
                     krylov_bases.get());
        */
        // print_matrix(std::string("gives_sin ") + std::to_string(total_iter),
        // krylov_bases.get());  print_matrix(std::string("gives_cos ") +
        // std::to_string(total_iter), krylov_bases.get());

        // for i in 0:restart_iter
        //     hessenberg(restart_iter, i) = next_krylov_basis' *
        //     krylov_bases(:, i) next_krylov_basis  -= hessenberg(restart_iter,
        //     i) * krylov_bases(:, i)
        // end
        // hessenberg(restart_iter, restart_iter + 1) = norm(next_krylov_basis)
        // next_krylov_basis /= hessenberg(restart_iter, restart_iter + 1)
        // End of arnoldi
        // Start apply givens rotation
        // for j in 0:restart_iter
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
        //                         cos(j)*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // Calculate sin and cos
        // hessenberg(restart_iter)   =
        // cos(restart_iter)*hessenberg(restart_iter) +
        //                      sin(restart_iter)*hessenberg(restart_iter)
        // hessenberg(restart_iter+1) = 0
        // End apply givens rotation
        // Calculate residual norm

        restart_iter++;
    }

    // Solve x
    auto krylov_bases_small = krylov_bases->create_submatrix(
        span{0, system_matrix_->get_size()[0]},
        span{0, dense_b->get_size()[1] * (restart_iter + 1)});
    auto hessenberg_small = hessenberg->create_submatrix(
        span{0, restart_iter},
        span{0, dense_b->get_size()[1] * (restart_iter)});

    if (false && exec->get_master() != exec) {
        auto h_rnc = Vector::create(exec->get_master());
        h_rnc->copy_from(residual_norm_collection.get());
        auto h_kbs = Vector::create(exec->get_master());
        h_kbs->copy_from(krylov_bases_small.get());
        auto h_hs = Vector::create(exec->get_master());
        h_hs->copy_from(hessenberg_small.get());
        auto h_y = Vector::create(exec->get_master());
        h_y->copy_from(y.get());
        auto h_x = Vector::create(exec->get_master());
        h_x->copy_from(dense_x);
        auto h_prec = matrix::Identity<ValueType>::create(
            exec->get_master(), preconditioner_->get_size()[0]);
        final_iter_nums.set_executor(exec->get_master());

        exec->get_master()->run(
            gmres::make_step_2(h_rnc.get(), h_kbs.get(), h_hs.get(), h_y.get(),
                               h_x.get(), &final_iter_nums, h_prec.get()));

        final_iter_nums.set_executor(exec);

        dense_x->copy_from(h_x.get());
        y->copy_from(h_y.get());
    } else {
        exec->run(gmres::make_step_2(residual_norm_collection.get(),
                                     krylov_bases_small.get(),
                                     hessenberg_small.get(), y.get(), dense_x,
                                     &final_iter_nums, preconditioner_.get()));
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
    // Solve x
    // x = x + preconditioner_ * krylov_bases * y

    print_matrix(std::string("Result y:"), y.get());
    print_matrix(std::string("Result x:"), dense_x);
    std::cout << "Total_iter: " << total_iter << '\n';
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *residual_norm_collection,
                                  LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
#undef GKO_DECLARE_GMRES


}  // namespace solver
}  // namespace gko
