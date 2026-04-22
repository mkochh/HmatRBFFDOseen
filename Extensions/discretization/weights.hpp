#ifndef SET_WEIGHTS_HEADER
#define SET_WEIGHTS_HEADER

//#include "../../medusa/include/medusa/Medusa.hpp"
#include "domain.hpp"
#include "../auxiliaries/aux_medusa.hpp"
#include <medusa/Medusa_fwd.hpp>
#include "../../medusa/include/medusa/bits/approximations/RBFFD.hpp"
#include "../../medusa/include/medusa/bits/approximations/Polyharmonic.hpp"
#include <Eigen/SparseCore>
#include <complex>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <array>

namespace mm {

// set weights for convection-diffusion operator, dirichlet and neumann boundary conditions possible
template<typename Operator1, typename Operator2>
void setWeightsCD3(Operator1& op_conv, Operator2& op_lap, const OseenDiscretizationBetter& dc)
{
    for (int i : dc.idxs_ui) {
        -dc.nu * op_lap.lap(i) = dc.rhs_vec3d[i];
        op_conv.grad(i,dc.conv_vec3d[i]) = 0;
    }
    for (int i : dc.idxs_dir) {
        op_lap.value(i) = dc.dir_bnd_vec3d[i];
    }
    for (int i : dc.idxs_neu) {
        dc.nu*op_conv.neumann(i, dc.d_u.normal(i), dc.idxs_u_ghost[i]) = dc.neu_bnd_vec3d[i];
    }
}

// set weights for convection-diffusion operator, dirichlet and neumann boundary conditions possible
template<typename Operator1, typename Operator2>
void setWeightsCD3(Operator1& op_conv, Operator2& op_lap, const OseenDiscretizationBetter& dc, const mm::Range<int>& idx_vel)
{
    for (int i : idx_vel) {
        -dc.nu * op_lap.lap(i) = dc.rhs_vec3d[i];
        op_conv.grad(i,dc.conv_vec3d[i]) = 0;
    }
    for (int i : dc.idxs_dir) {
        op_lap.value(i) = dc.dir_bnd_vec3d[i];
    }
    for (int i : dc.idxs_neu) {
        dc.nu*op_conv.neumann(i, dc.d_u.normal(i), dc.idxs_u_ghost[i]) = dc.neu_bnd_vec3d[i];
    }
}

// set weights for convection-diffusion operator, dirichlet and neumann boundary conditions possible
template<typename Operator1, typename Operator2>
void setWeightsCD(Operator1& op_conv, Operator2& op_lap, const OseenDiscretizationBetter& dc, const mm::Range<int>& idx_vel, const mm::Range<int>& idx_neu)
{
    for (int i : idx_vel) {
        -dc.nu * op_lap.lap(i) = 0;
        op_conv.grad(i,dc.conv_vec3d[i]) = 0;
    }
    for (int i : dc.idxs_dir) {
        op_lap.value(i) = 0;
    }
    for (int i : idx_neu) {
        dc.nu*op_conv.neumann(i, dc.d_u.normal(i), dc.idxs_u_ghost[i]) = 0;
    }
}

template<typename Operator>
void setWeightsDIV3(Operator& op_div, int N_u, const Range<int>& idxs_p)
{
    op_div.setRowOffset(2*N_u);
    for (int i : idxs_p) {
        op_div.der1(i,0) = 0;
    }
    op_div.setColOffset(N_u);
    for (int i : idxs_p) {
        op_div.der1(i,1) = 0;
    }
    op_div.setColOffset(2*N_u);
    for (int i : idxs_p) {
        op_div.der1(i,2) = 0;
    }
}

template<typename Operator>
void setWeightsDIV(Operator& op_div, const Range<int>& idxs_p, int dim)
{
    for (int i : idxs_p) {
        op_div.der1(i,dim) = 0;
    }
}

// Represents the Value operator
template <int dim>
struct Value : public Operator<Value<dim>> {
    typedef Vec<double, dim> vec;
    static std::string type_name() { return format("Value<%d>", dim); }
    static std::string name() { return type_name(); }

    template <int k>
    double applyAt0(const RBFBasis<Polyharmonic<double, k>, vec>& basis, int index,
                    const std::vector<vec>& support, double) const {
        // static_assert(k != -1, "If dynamic k is desired it can be obtained from basis.rbf().");
        const int order = basis.rbf().order(); // k might be -1 if order is assigned dynamically, therefore find order like this
        double r = support[index].norm();
        return ipow(r, order);
    }

    double applyAt0(const Monomials<vec>& mon, int idx, const std::vector<vec>& q, double) const {
        return mon.evalAt0(idx, q);
    }
};

// set weights for gradient operator, supports neumann and dirichlet boundary conditions
template<typename Operator1, typename Operator2>
void setWeightsGRAD3(Operator1& op_grad, Operator2& op_value, const OseenDiscretizationBetter& dc)
{
    op_grad.setColOffset(2*dc.N_u);
    op_value.setColOffset(2*dc.N_u);
    for (int i : dc.idxs_ui) {
        op_grad.der1(i,0) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[0]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }

    op_grad.setRowOffset(dc.N_u);
    op_value.setRowOffset(dc.N_u);
    for (int i : dc.idxs_ui) {
        op_grad.der1(i,1) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[1]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }

    op_grad.setRowOffset(2*dc.N_u);
    op_value.setRowOffset(2*dc.N_u);
    for (int i : dc.idxs_ui) {
        op_grad.der1(i,2) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[2]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }
}

// set weights for gradient operator, supports neumann and dirichlet boundary conditions
template<typename Operator1, typename Operator2>
void setWeightsGRAD3(Operator1& op_grad, Operator2& op_value, const OseenDiscretizationBetter& dc, const mm::Range<int> idx_grad)
{
    op_grad.setColOffset(2*dc.N_u);
    op_value.setColOffset(2*dc.N_u);
    for (int i : idx_grad) {
        op_grad.der1(i,0) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[0]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }

    op_grad.setRowOffset(dc.N_u);
    op_value.setRowOffset(dc.N_u);
    for (int i : idx_grad) {
        op_grad.der1(i,1) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[1]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }

    op_grad.setRowOffset(2*dc.N_u);
    op_value.setRowOffset(2*dc.N_u);
    for (int i : idx_grad) {
        op_grad.der1(i,2) = 0;
    }
    for (int i : dc.idxs_neu) {
        -dc.d_u.normal(i)[2]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }
}

// set weights for gradient operator, supports neumann and dirichlet boundary conditions
template<typename Operator1, typename Operator2>
void setWeightsGRAD(Operator1& op_grad, Operator2& op_value, const OseenDiscretizationBetter& dc, const mm::Range<int> idx_grad, 
    const mm::Range<int> idx_neu, int dim)
{
    for (int i : idx_grad) {
        op_grad.der1(i,dim) = 0;
    }
    for (int i : idx_neu) {
        -dc.d_u.normal(i)[dim]*op_value.template apply<Value<3>>(i, {}, dc.idxs_u_ghost[i]) = 0;
    }
}

void setPressureConstraint3(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_with_dirbnd, int N_u, int N_p, const Eigen::VectorXd &constraint)
{
    for(int i = 3*N_u; i < 3*N_u+N_p; i++) {
        mat_with_dirbnd.insert(3*N_u+N_p,i) = constraint(i-3*N_u);
        mat_with_dirbnd.insert(i,3*N_u+N_p) = constraint(i-3*N_u);
    }
}

void setPressureConstraint(Eigen::SparseMatrix<double, Eigen::RowMajor>& schur, const Eigen::VectorXd &constraint)
{
    for(int i = 0; i < schur.rows() - 1; i++) {
        schur.insert(schur.rows() - 1, i) = constraint(i);
        schur.insert(i, schur.rows() - 1) = constraint(i);
    }
}

template<typename Storage>
double computeGrowthFactor(const OseenDiscretizationBetter &dc, Storage &storage, double eps) {
    // eps is real part of eigenvalue with largest real part
    double khat = 2/dc.dx_u; // wave number
    double q = 0.0; // growth factor
    
    Eigen::VectorXcd gtilde(dc.N_u), f(dc.N_u); gtilde.setZero(); f.setZero();
    const std::complex<double> imag(0.0, 1.0);

    Range<Eigen::VectorXcd> g(3, Eigen::VectorXcd(dc.N_u));

    // f = exp(i*khat*(x*y*z))
    for (int i : dc.d_u.all()) {
        double x = dc.d_u.pos(i,0);
        double y = dc.d_u.pos(i,1);
        double z = dc.d_u.pos(i,2);
        f(i) = std::exp(imag*khat*(x+y+z));
        g[0](i) = imag*khat*std::exp(imag*khat*x);
        g[1](i) = imag*khat*std::exp(imag*khat*y);
        g[2](i) = imag*khat*std::exp(imag*khat*z);
    }

    Range<Vec3d> conv(3);
    conv[0] = {1.0, 0.0, 0.0};
    conv[1] = {0.0, 1.0, 0.0};
    conv[2] = {0.0, 0.0, 1.0};

    // compute growth factor for all 3 directions and take the maximum
    for (int i = 0; i < 3; i++) {
        Eigen::SparseMatrix<double, Eigen::RowMajor> G(dc.N_u,dc.N_u); // derivative approximation
        Eigen::VectorXd temp(dc.N_u); temp.setZero();
        Eigen::VectorXi reserve_vector(dc.N_u);
        reserve_vector << Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,dc.n[dc.poly_conv]);
        G.reserve(reserve_vector);
        auto op_conv = storage.implicitOperators(G, temp);
        for (int j : dc.idxs_ui) {
            op_conv.grad(j,conv[i]) = 0;
        }
        for (int i : dc.idxs_neu) {
            op_conv.neumann(i, dc.d_u.normal(i), dc.idxs_u_ghost[i]) = 0;
        }

        gtilde = G*f;
        double q_new = (std::log((gtilde-g[i]).norm()) - std::log(eps*f.norm()))/std::log(khat);
        q = std::max(q, q_new);
    }
    prn(q);

    return q;
}

template<typename Storage>
double computeEVwithLargestRealPart(const OseenDiscretizationBetter &dc, Storage &storage) {
    // create matrix with weigths for convetion
    Eigen::SparseMatrix<double, Eigen::RowMajor> M_full(dc.N_u,dc.N_u);
    Eigen::SparseMatrix<double, Eigen::RowMajor> M(dc.N_ui+dc.N_uneu,dc.N_ui+dc.N_uneu);
    Eigen::VectorXd temp(dc.N_u); temp.setZero();

    Eigen::VectorXi reserve_vector(dc.N_u);
    reserve_vector << Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,dc.n[dc.poly_conv]);
    M_full.reserve(reserve_vector);

    auto op_conv = storage.implicitOperators(M_full,temp);
    //#pragma omp parallel for
    for (int i : dc.idxs_ui) {
        op_conv.grad(i,dc.conv_vec3d[i]) = 0;
    }
    for (int i : dc.idxs_neu) {
        op_conv.neumann(i, dc.d_u.normal(i), dc.idxs_u_ghost[i]) = 0;
    }
    M_full.makeCompressed();

    // cut out dirichlet boundary part
    M = M_full.block(dc.N_ub,dc.N_ub,dc.N_ui,dc.N_ui);

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    Spectra::SparseGenMatProd<double, Eigen::RowMajor> op(M);

    // Construct eigen solver object, requesting the eigenvalue with the largest real part
    Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double, Eigen::RowMajor>> eigs(op, 1, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestReal, 1000, 1e-2);

    // Retrieve results
    Eigen::VectorXcd evalues;
    if(eigs.info() == Spectra::CompInfo::Successful)
        evalues = eigs.eigenvalues();

    double eps = 0.0;
    if (nconv != 0) {
        eps = evalues[0].real();
        prn(eps);
        if (eps < 0 || nconv == 0) {
            std::cout << "Real part of eigenvalue with largest real part is negative, setting to 0" << std::endl;
            eps = 0;
        }
    } else {
        std::cout << "Eigenvalue computation did not converge, setting to 0" << std::endl;
        eps = 0;
    }
    return eps;
}

// operator for laplace^(k-1)/2, assumes that highest degree of polynomial is smaller than (k-1)/2
template <int dim>
struct Hyperviscosity : public Operator<Hyperviscosity<dim>> {
    typedef Vec<double, dim> vec;
    static std::string type_name() { return format("Hyperviscosity<%d>", dim); }
 
    template <int k>
    double applyAt0(const RBFBasis<Polyharmonic<double, k>, vec>& basis, int index,
                    const std::vector<vec>& support, double scale) const {
        // static_assert(k != -1, "If dynamic k is desired it can be obtained from basis.rbf().");
        const int order = basis.rbf().order(); // k might be -1 if order is assigned dynamically, therefore find order like this
        double r = support[index].norm();
        double result = r;
        for (int i = 0; i < (order-1)/2; i++) {
            result *= (2*i+1)*(2*i+1+dim-2);
        }
        
        return result / ipow(scale, order);
    }
 
    double applyAt0(const Monomials<vec>&, int, const std::vector<vec>&, double) const {
        return 0.0;
    }
};

void computeHyperviscosity(const OseenDiscretizationBetter &dc, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat, double q, double eps) {
    int l_hyp = dc.poly_hyp; // polynomial augmentation degree for hyperviscosity
    int k_hyp = dc.k_hyp; // degree of polyharmonic spline for hyperviscosity
    int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
    Eigen::VectorXd temp(3*dc.N_u+dc.N_p+1); temp.setZero();

    int k_ = (k_hyp-1)/2;
    // double max_conv = 0.0;
    // for (int i = 0; i < dc.conv_vec3d.size(); i++) {
    //     max_conv = std::max(max_conv, dc.conv_vec3d[i].lpNorm<Eigen::Infinity>());
    // }
    // prn(max_conv);

    // max_conv local worked, while global sometimes failed for most cases
    Eigen::VectorXd max_conv = Eigen::VectorXd::Zero(dc.conv_vec3d.size());
    for (int i = 0; i < dc.conv_vec3d.size(); i++) {
        max_conv[i] = dc.conv_vec3d[i].lpNorm<Eigen::Infinity>();
    }
    prn(max_conv.maxCoeff());

    // double gamma = std::pow(-1,1-k_)*std::pow(2,q-2*k_)*std::pow(dc.dx_u,2*k_-q)*max_conv*eps; // hyperviscosity coefficient
    double gamma = std::pow(-1,1-k_)*std::pow(2,q-2*k_)*std::pow(dc.dx_u,2*k_-q)*eps; // hyperviscosity coefficient
    prn(gamma);

    prn(l_hyp);
    prn(k_hyp);
    prn(n_hyp);

    Polyharmonic<double, -1> ph_hyp(k_hyp);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_hyp(ph_hyp, l_hyp);

    std::tuple<Hyperviscosity<3>> ops;
    RaggedShapeStorage<Vec3d, decltype(ops)> storage_hyp;
    storage_hyp.resize(dc.d_hyp.supportSizes());
    computeShapes(dc.d_hyp, approx_hyp, dc.idxs_ui, std::tuple<Hyperviscosity<3>>(), &storage_hyp);
    auto op_hyp = storage_hyp.implicitVectorOperators(mat, temp);

    for (int i : dc.idxs_ui) {
        -gamma * max_conv[i-dc.idxs_u[0]] * op_hyp.apply<Hyperviscosity<3>>(i) = 0;
    }
}

void computeHyperviscosity(const OseenDiscretizationBetter &dc, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat, double q, double eps,
    mm::Range<int> idxs_vel) {
    int l_hyp = dc.poly_hyp; // polynomial augmentation degree for hyperviscosity
    int k_hyp = dc.k_hyp; // degree of polyharmonic spline for hyperviscosity
    int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
    Eigen::VectorXd temp(mat.rows()); temp.setZero();

    int k_ = (k_hyp-1)/2;
    // double max_conv = 0.0;
    // for (int i = 0; i < dc.conv_vec3d.size(); i++) {
    //     max_conv = std::max(max_conv, dc.conv_vec3d[i].lpNorm<Eigen::Infinity>());
    // }
    // prn(max_conv);

    // max_conv local worked, while global sometimes failed for most cases
    Eigen::VectorXd max_conv = Eigen::VectorXd::Zero(dc.conv_vec3d.size());
    for (int i = 0; i < dc.conv_vec3d.size(); i++) {
        max_conv[i] = dc.conv_vec3d[i].lpNorm<Eigen::Infinity>();
    }
    prn(max_conv.maxCoeff());

    // double gamma = std::pow(-1,1-k_)*std::pow(2,q-2*k_)*std::pow(dc.dx_u,2*k_-q)*max_conv*eps; // hyperviscosity coefficient
    double gamma = std::pow(-1,1-k_)*std::pow(2,q-2*k_)*std::pow(dc.dx_u,2*k_-q)*eps; // hyperviscosity coefficient
    prn(gamma);

    prn(l_hyp);
    prn(k_hyp);
    prn(n_hyp);

    Polyharmonic<double, -1> ph_hyp(k_hyp);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_hyp(ph_hyp, l_hyp);

    std::tuple<Hyperviscosity<3>> ops;
    RaggedShapeStorage<Vec3d, decltype(ops)> storage_hyp;
    storage_hyp.resize(dc.d_hyp.supportSizes());
    computeShapes(dc.d_hyp, approx_hyp, idxs_vel, std::tuple<Hyperviscosity<3>>(), &storage_hyp);
    auto op_hyp = storage_hyp.implicitVectorOperators(mat, temp);

    for (int i : idxs_vel) {
        -gamma * max_conv[i-dc.idxs_u[0]] * op_hyp.apply<Hyperviscosity<3>>(i) = 0;
    }
}

template<typename Storage>
void addHyperviscosity(const OseenDiscretizationBetter &dc, Storage &storage, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd) {
    double eps = computeEVwithLargestRealPart(dc, storage);
    if (eps > 0) {
        double q = computeGrowthFactor(dc, storage, eps);
        computeHyperviscosity(dc, mat_with_dirbnd, q, eps);
    }
}

template<typename Storage>
void addHyperviscosity(const OseenDiscretizationBetter &dc, Storage &storage, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, 
    mm::Range<int> idxs_vel) {
    double eps = computeEVwithLargestRealPart(dc, storage);
    if (eps > 0) {
        double q = computeGrowthFactor(dc, storage, eps);
        computeHyperviscosity(dc, mat_with_dirbnd, q, eps, idxs_vel);
    }
}

// set weights for dirichlet and neumann boundary conditions
Eigen::VectorXd setWeightsOseen(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, const OseenDiscretizationBetter &dc)
{
    int l_lap = dc.l[dc.poly_lap], l_grad = dc.l[dc.poly_grad], l_conv = dc.l[dc.poly_conv], l_div = dc.l[dc.poly_div];
    int k_lap = dc.k[dc.poly_lap], k_grad = dc.k[dc.poly_grad], k_conv = dc.k[dc.poly_conv], k_div = dc.k[dc.poly_div];
    int n1 = dc.n[dc.poly_grad], n2 = std::max(dc.n[dc.poly_lap],dc.n[dc.poly_conv]), n3 = dc.n[dc.poly_div];
    if (dc.use_hyperviscosity) { 
        int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
        n2 = std::max(n2, n_hyp);
    }
    Eigen::VectorXi reserve_vector(dc.N_all);
    // this overestimates the number of non-zeros slightly, but this is fine
    reserve_vector << Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_with_dirbnd.reserve(reserve_vector);
    Eigen::VectorXd rhs_with_dirbnd(dc.N_all); rhs_with_dirbnd.setZero();
    Eigen::VectorXd rhs_dummy(dc.N_all); rhs_dummy.setZero(); // dummy vector, discarded later
    
    Polyharmonic<double, -1> ph_lap(k_lap);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_lap(ph_lap, l_lap);
    auto storage_lap = dc.d_u.computeShapes<sh::lap>(approx_lap, dc.idxs_ui);
    auto op_lap = storage_lap.implicitVectorOperators(mat_with_dirbnd, rhs_with_dirbnd);

    Polyharmonic<double, -1> ph_conv(k_conv);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_conv(ph_conv, l_conv);
    auto storage_conv = dc.d_conv.computeShapes<sh::d1>(approx_conv, dc.idxs_ui);
    auto op_conv = storage_conv.implicitVectorOperators(mat_with_dirbnd,rhs_with_dirbnd);
    
    Polyharmonic<double, -1> ph_grad(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_grad(ph_grad, l_grad);
    auto storage_grad = dc.d_grad.computeShapes<sh::d1>(approx_grad, dc.idxs_ui);
    auto op_grad = storage_grad.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx(ph, l_grad);
    std::tuple<Value<3>> ops;
    RaggedShapeStorage<Vec3d, decltype(ops)> storage_value;
    storage_value.resize(dc.d_grad.supportSizes());
    computeShapes(dc.d_grad, approx, dc.idxs_neu, std::tuple<Value<3>>(), &storage_value);
    auto op_value = storage_value.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph_div(k_div);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_div(ph_div, l_div);
    auto storage_div = dc.d_div.computeShapes<sh::d1>(approx_div, dc.idxs_p);
    auto op_div = storage_div.implicitOperators(mat_with_dirbnd, rhs_dummy);

    setWeightsCD3(op_conv, op_lap, dc);
    setWeightsDIV3(op_div, dc.N_u, dc.idxs_p);
    setWeightsGRAD3(op_grad, op_value, dc);
    setPressureConstraint3(mat_with_dirbnd, dc.N_u, dc.N_p, dc.constraint);

    if (dc.use_hyperviscosity) {
        addHyperviscosity(dc, storage_conv, mat_with_dirbnd);
    }

    return rhs_with_dirbnd;
}

// set weights for dirichlet and neumann boundary conditions, indices specified in idx
// useful for updating only certain stencils
void setWeightsOseen(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_with_dirbnd, const OseenDiscretizationBetter& dc, const Indices& idx)
{
    int l_lap = dc.l[dc.poly_lap], l_grad = dc.l[dc.poly_grad], l_conv = dc.l[dc.poly_conv], l_div = dc.l[dc.poly_div];
    int k_lap = dc.k[dc.poly_lap], k_grad = dc.k[dc.poly_grad], k_conv = dc.k[dc.poly_conv], k_div = dc.k[dc.poly_div];
    int n1 = dc.n[dc.poly_grad], n2 = std::max(dc.n[dc.poly_lap],dc.n[dc.poly_conv]), n3 = dc.n[dc.poly_div];
    if (dc.use_hyperviscosity) { 
        int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
        n2 = std::max(n2, n_hyp);
    }
    Eigen::VectorXi reserve_vector(dc.N_all);
    // this overestimates the number of non-zeros slightly, but this is fine
    reserve_vector << Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_with_dirbnd.reserve(reserve_vector);
    Eigen::VectorXd rhs_dummy(dc.N_all); rhs_dummy.setZero(); // dummy vector, discarded later
    
    Polyharmonic<double, -1> ph_lap(k_lap);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_lap(ph_lap, l_lap);
    auto storage_lap = dc.d_u.computeShapes<sh::lap>(approx_lap, idx.idxs_vel);
    auto op_lap = storage_lap.implicitVectorOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph_conv(k_conv);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_conv(ph_conv, l_conv);
    auto storage_conv = dc.d_conv.computeShapes<sh::d1>(approx_conv, idx.idxs_vel);
    auto op_conv = storage_conv.implicitVectorOperators(mat_with_dirbnd, rhs_dummy);
    
    Polyharmonic<double, -1> ph_grad(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_grad(ph_grad, l_grad);
    auto storage_grad = dc.d_grad.computeShapes<sh::d1>(approx_grad, idx.idxs_grad);
    auto op_grad = storage_grad.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx(ph, l_grad);
    std::tuple<Value<3>> ops;
    RaggedShapeStorage<Vec3d, decltype(ops)> storage_value;
    storage_value.resize(dc.d_grad.supportSizes());
    computeShapes(dc.d_grad, approx, dc.idxs_neu, std::tuple<Value<3>>(), &storage_value);
    auto op_value = storage_value.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph_div(k_div);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_div(ph_div, l_div);
    auto storage_div = dc.d_div.computeShapes<sh::d1>(approx_div, idx.idxs_div);
    auto op_div = storage_div.implicitOperators(mat_with_dirbnd, rhs_dummy);

    setWeightsCD3(op_conv, op_lap, dc, idx.idxs_vel);
    setWeightsDIV3(op_div, dc.N_u, idx.idxs_div);
    setWeightsGRAD3(op_grad, op_value, dc, idx.idxs_grad);
    setPressureConstraint3(mat_with_dirbnd, dc.N_u, dc.N_p, dc.constraint);

    if (dc.use_hyperviscosity) {
        addHyperviscosity(dc, storage_conv, mat_with_dirbnd);
    }
}

class OseenMatrix {
    public:
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_vel;
    std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, 3> mat_grad;
    std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, 3> mat_div;
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_schur;

    OseenMatrix() = default;

    OseenMatrix(int N_u, int N_p) : mat_vel(N_u, N_u), mat_schur(N_p+1, N_p+1) {
        for (int i = 0; i < 3; i++) {
            mat_grad[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>(N_u, N_p+1);
            mat_div[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>(N_p+1, N_u);
        }
    }

    OseenMatrix(int N_u, int N_p, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat) {
        assert(3*N_u + N_p + 1 == mat.rows());
        assert(3*N_u + N_p + 1 == mat.cols());

        mat_vel = mat.block(0, 0, N_u, N_u);
        mat_schur = mat.block(3*N_u, 3*N_u, N_p+1, N_p+1);
        for (int i = 0; i < 3; i++) {
            mat_grad[i] = mat.block(i*N_u, 3*N_u, N_u, N_p+1);
            mat_div[i] = mat.block(3*N_u, i*N_u, N_p+1, N_u);
        }
    }
};

// set weights for dirichlet and neumann boundary conditions, indices specified in idx
// useful for updating only certain stencils
OseenMatrix setWeightsOseen(const OseenDiscretizationBetter& dc, const Indices& idx)
{
    int l_lap = dc.l[dc.poly_lap], l_grad = dc.l[dc.poly_grad], l_conv = dc.l[dc.poly_conv], l_div = dc.l[dc.poly_div];
    int k_lap = dc.k[dc.poly_lap], k_grad = dc.k[dc.poly_grad], k_conv = dc.k[dc.poly_conv], k_div = dc.k[dc.poly_div];
    int n1 = dc.n[dc.poly_grad], n2 = std::max(dc.n[dc.poly_lap],dc.n[dc.poly_conv]), n3 = dc.n[dc.poly_div];
    if (dc.use_hyperviscosity) { 
        int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
        n2 = std::max(n2, n_hyp);
    }

    OseenMatrix oseen_matrix(dc.N_u, dc.N_p);

    Eigen::VectorXi reserve_vector_vel(dc.N_u);
    Eigen::VectorXi reserve_vector_grad(dc.N_u);
    Eigen::VectorXi reserve_vector_div(dc.N_p+1);
    Eigen::VectorXi reserve_vector_schur(dc.N_p+1);
    // this overestimates the number of non-zeros slightly, but this is fine
    reserve_vector_vel << Eigen::VectorXi::Constant(dc.N_ub,1), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu, n2);
    reserve_vector_grad << Eigen::VectorXi::Constant(dc.N_ub,0), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu, n1);
    reserve_vector_div << Eigen::VectorXi::Constant(dc.N_p,3*n3), Eigen::VectorXi::Constant(1, 0);
    reserve_vector_schur << Eigen::VectorXi::Constant(dc.N_p,1), Eigen::VectorXi::Constant(1, dc.N_p);
    oseen_matrix.mat_vel.reserve(reserve_vector_vel);
    oseen_matrix.mat_schur.reserve(reserve_vector_schur);
    for (int i = 0; i < 3; i++) {
        oseen_matrix.mat_grad[i].reserve(reserve_vector_grad);
        oseen_matrix.mat_div[i].reserve(reserve_vector_div);
    }

    Eigen::VectorXd rhs_dummy(dc.N_u + dc.N_p); rhs_dummy.setZero(); // dummy vector, discarded later

    if (idx.idxs_vel.size() > 0) { // this is necessary because medusa uses all idx in the domain if the idx given are empty
        Polyharmonic<double, -1> ph_lap(k_lap);
        RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_lap(ph_lap, l_lap);
        auto storage_lap = dc.d_u.computeShapes<sh::lap>(approx_lap, idx.idxs_vel);
        auto op_lap = storage_lap.implicitOperators(oseen_matrix.mat_vel, rhs_dummy);

        Polyharmonic<double, -1> ph_conv(k_conv);
        RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToClosest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_conv(ph_conv, l_conv);
        if (dc.use_hyperviscosity) {
            // need to compute storage_conv for dc.idxs_ui to be able to compute hyperviscosity
            auto storage_conv = dc.d_conv.computeShapes<sh::d1>(approx_conv, dc.idxs_ui);
            auto op_conv = storage_conv.implicitOperators(oseen_matrix.mat_vel, rhs_dummy);

            setWeightsCD(op_conv, op_lap, dc, idx.idxs_vel, idx.idxs_neu_vel);

            addHyperviscosity(dc, storage_conv, oseen_matrix.mat_vel, idx.idxs_vel);
        } else {
            auto storage_conv = dc.d_conv.computeShapes<sh::d1>(approx_conv, idx.idxs_vel);
            auto op_conv = storage_conv.implicitOperators(oseen_matrix.mat_vel, rhs_dummy);

            setWeightsCD(op_conv, op_lap, dc, idx.idxs_vel, idx.idxs_neu_vel);
        }   
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> temp_mat(dc.d_grad.size(), dc.d_grad.size()); // hack to get around assert in implicitOperators
    
    if (idx.idxs_grad.size() > 0) {
        Polyharmonic<double, -1> ph_grad(k_grad);
        RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_grad(ph_grad, l_grad);
        auto storage_grad = dc.d_grad.computeShapes<sh::d1>(approx_grad, idx.idxs_grad);
        auto op_grad = storage_grad.implicitOperators(temp_mat, rhs_dummy);

        Polyharmonic<double, -1> ph(k_grad);
        RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx(ph, l_grad);
        std::tuple<Value<3>> ops;
        RaggedShapeStorage<Vec3d, decltype(ops)> storage_value;
        storage_value.resize(dc.d_grad.supportSizes());
        computeShapes(dc.d_grad, approx, idx.idxs_neu_grad, std::tuple<Value<3>>(), &storage_value);
        auto op_value = storage_value.implicitOperators(temp_mat, rhs_dummy);

        for (int i = 0; i < 3; i++) {
            // have to comment out some asserts in implcitOperators to allow for negative row/column offsets
            op_grad.setProblem(oseen_matrix.mat_grad[i], rhs_dummy, 0, -dc.d_u.size());
            op_value.setProblem(oseen_matrix.mat_grad[i], rhs_dummy, 0, -dc.d_u.size());
            setWeightsGRAD(op_grad, op_value, dc, idx.idxs_grad, idx.idxs_neu_grad, i);
        }
    }
    
    if (idx.idxs_div.size() > 0) {
        Polyharmonic<double, -1> ph_div(k_div);
        RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_div(ph_div, l_div);
        auto storage_div = dc.d_div.computeShapes<sh::d1>(approx_div, idx.idxs_div);
        auto op_div = storage_div.implicitOperators(temp_mat, rhs_dummy);

        for (int i = 0; i < 3; i++) {
            // have to comment out some asserts in implcitOperators to allow for negative row/column offsets
            op_div.setProblem(oseen_matrix.mat_div[i], rhs_dummy, -dc.d_u.size(), 0);
            setWeightsDIV(op_div, idx.idxs_div, i);
        }
    }

    setPressureConstraint(oseen_matrix.mat_schur, dc.constraint);

    return oseen_matrix;
}

// set weights for dirichlet and neumann boundary conditions
void setWeightsGradDiv(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, const OseenDiscretizationBetter &dc)
{
    int l_grad = dc.l[dc.poly_grad], l_div = dc.l[dc.poly_div];
    int k_grad = dc.k[dc.poly_grad], k_div = dc.k[dc.poly_div];
    int n1 = dc.n[dc.poly_grad], n3 = dc.n[dc.poly_div];
    Eigen::VectorXi reserve_vector(dc.N_all);
    // this overestimates the number of non-zeros slightly, but this is fine
    reserve_vector << Eigen::VectorXi::Constant(dc.N_ub,0), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,0), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_ub,0), Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_with_dirbnd.reserve(reserve_vector);
    Eigen::VectorXd rhs_dummy(dc.N_all); rhs_dummy.setZero(); // dummy vector, discarded later
    
    Polyharmonic<double, -1> ph_grad(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_grad(ph_grad, l_grad);
    auto storage_grad = dc.d_grad.computeShapes<sh::d1>(approx_grad, dc.idxs_ui);
    auto op_grad = storage_grad.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph(k_grad);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx(ph, l_grad);
    std::tuple<Value<3>> ops;
    RaggedShapeStorage<Vec3d, decltype(ops)> storage_value;
    storage_value.resize(dc.d_grad.supportSizes());
    computeShapes(dc.d_grad, approx, dc.idxs_neu, std::tuple<Value<3>>(), &storage_value);
    auto op_value = storage_value.implicitOperators(mat_with_dirbnd, rhs_dummy);

    Polyharmonic<double, -1> ph_div(k_div);
    RBFFD<Polyharmonic<double, -1>, Vec3d, ScaleToFarthest, Eigen::PartialPivLU<Eigen::MatrixXd>> approx_div(ph_div, l_div);
    auto storage_div = dc.d_div.computeShapes<sh::d1>(approx_div, dc.idxs_p);
    auto op_div = storage_div.implicitOperators(mat_with_dirbnd, rhs_dummy);

    setWeightsDIV3(op_div, dc.N_u, dc.idxs_p);
    setWeightsGRAD3(op_grad, op_value, dc);
    setPressureConstraint3(mat_with_dirbnd, dc.N_u, dc.N_p, dc.constraint);
}

// remove part in matrix corresponding to boundary conditions
Eigen::VectorXd eliminateDirichlet(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_without_dirbnd, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, 
                                    const OseenDiscretizationBetter &dc, Eigen::VectorXd &rhs_with_dirbnd)
{
    int n1 = dc.n[dc.poly_grad], n2 = std::max(dc.n[dc.poly_lap],dc.n[dc.poly_conv]), n3 = dc.n[dc.poly_div];
    if (dc.use_hyperviscosity) { 
        int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
        n2 = std::max(n2, n_hyp);
    }
    Eigen::VectorXi reserve_vector_mat(dc.N_dofs);
    reserve_vector_mat << Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_without_dirbnd.reserve(reserve_vector_mat);
    Eigen::SparseMatrix<double, Eigen::RowMajor> T; // temporarily stores blocks of mat_with_dirbnd in function reduceMat, not needed afterwards
    reduceMat3(mat_without_dirbnd, mat_with_dirbnd, T, dc.N_u, dc.N_p, dc.N_ui+dc.N_uneu, dc.N_ub, dc.constraint);
    mat_without_dirbnd.makeCompressed();

    Eigen::VectorXd rhs_without_dirbnd(dc.N_dofs); rhs_without_dirbnd.setZero();
    Range<int> idxs_dofs = dc.idxs_ui;
    idxs_dofs.append(dc.idxs_u_ghost_global);
    adjustRHS3(idxs_dofs, rhs_with_dirbnd, rhs_without_dirbnd, mat_with_dirbnd, dc.N_u, dc.N_ub, dc.N_p); // adjust rhs accordingly

    return rhs_without_dirbnd;
}

// remove part in matrix corresponding to boundary conditions
void eliminateDirichlet(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_without_dirbnd, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, 
                                    const OseenDiscretizationBetter &dc)
{
    int n1 = dc.n[dc.poly_grad], n2 = std::max(dc.n[dc.poly_lap],dc.n[dc.poly_conv]), n3 = dc.n[dc.poly_div];
    if (dc.use_hyperviscosity) { 
        int n_hyp = dc.n_hyp; // stencil size for hyperviscosity
        n2 = std::max(n2, n_hyp);
    }
    Eigen::VectorXi reserve_vector_mat(dc.N_dofs);
    reserve_vector_mat << Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n2+n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_without_dirbnd.reserve(reserve_vector_mat);
    Eigen::SparseMatrix<double, Eigen::RowMajor> T; // temporarily stores blocks of mat_with_dirbnd in function reduceMat, not needed afterwards
    reduceMat3(mat_without_dirbnd, mat_with_dirbnd, T, dc.N_u, dc.N_p, dc.N_ui+dc.N_uneu, dc.N_ub, dc.constraint);
    mat_without_dirbnd.makeCompressed();
}

OseenMatrix eliminateDirichlet(const OseenMatrix& mat_with_dirbnd, const OseenDiscretizationBetter& dc)
{
    OseenMatrix mat_without_dirbnd;

    mat_without_dirbnd.mat_vel = mat_with_dirbnd.mat_vel.block(dc.N_ub, dc.N_ub, dc.N_ui + dc.N_uneu, dc.N_ui + dc.N_uneu);
    for (int i = 0; i < 3; i++) {
        mat_without_dirbnd.mat_grad[i] = mat_with_dirbnd.mat_grad[i].block(dc.N_ub, 0, dc.N_ui + dc.N_uneu, dc.N_p + 1);
        mat_without_dirbnd.mat_div[i] = mat_with_dirbnd.mat_div[i].block(0, dc.N_ub, dc.N_p + 1, dc.N_ui + dc.N_uneu);
    }
    mat_without_dirbnd.mat_schur = mat_with_dirbnd.mat_schur;

    mat_without_dirbnd.mat_vel.makeCompressed();
    for (int i = 0; i < 3; i++) {
        mat_without_dirbnd.mat_grad[i].makeCompressed();
        mat_without_dirbnd.mat_div[i].makeCompressed();
    }
    mat_without_dirbnd.mat_schur.makeCompressed();

    return mat_without_dirbnd;
}

// remove part in matrix corresponding to boundary conditions
void eliminateDirichletGradDiv(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_without_dirbnd, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat_with_dirbnd, 
                                    const OseenDiscretizationBetter &dc)
{
    int n1 = dc.n[dc.poly_grad], n3 = dc.n[dc.poly_div];
    Eigen::VectorXi reserve_vector_mat(dc.N_dofs);
    reserve_vector_mat << Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_ui+dc.N_uneu,n1), 
                        Eigen::VectorXi::Constant(dc.N_p,3*n3+1), Eigen::VectorXi::Constant(1,dc.N_p);
    mat_without_dirbnd.reserve(reserve_vector_mat);
    Eigen::SparseMatrix<double, Eigen::RowMajor> T; // temporarily stores blocks of mat_with_dirbnd in function reduceMat, not needed afterwards
    reduceMat3(mat_without_dirbnd, mat_with_dirbnd, T, dc.N_u, dc.N_p, dc.N_ui+dc.N_uneu, dc.N_ub, dc.constraint);
    mat_without_dirbnd.makeCompressed();
}

// create Matrix and RHS with Dirichlet boundary nodes eliminated
void createMatrixAndRHS(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, Eigen::VectorXd& rhs, const OseenDiscretizationBetter& dc, 
    PressureConstraint pressure_constraint) {
    // full system and righthand-side including dirichlet boundary conditions
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_temp(dc.N_all, dc.N_all);
    Eigen::VectorXd rhs_temp(dc.N_all);
    rhs_temp = setWeightsOseen(mat_temp, dc);

    // reduced system without dirichlet boundary conditions
    rhs = eliminateDirichlet(mat, mat_temp, dc, rhs_temp);

    if (pressure_constraint == PressureConstraint::SET && dc.idxs_neu.size() == 0)
    {
        // set last 2 rows and last column to zero and set 2x2 block to identity, removes pressure constraint
        // and sets pressure at node N_p to value of exact pressure solution at node N_p (N_p corresponds to N_dofs-2 overall)
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-3 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-2,dc.N_dofs-2) = 1.0;
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        rhs(dc.N_dofs-2) = dc.exact_solution(dc.N_dofs-2); // set exact pressure solution at pressure node N_p (overall node N_dofs-2)
        mat.makeCompressed();
    }

    // don't need pressure constraint if Neumann boundary conditions are present
    if (dc.idxs_neu.size() > 0) {
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-2 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        mat.makeCompressed();
    }
}

// create Matrix that only contains values in rows with given Indices (and Neumann boundary, this could be optimized) 
// with Dirichlet boundary nodes eliminated
// this is useful for restricted stencil preconditioner to avoid recomputing the entire matrix
void createMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const OseenDiscretizationBetter& dc, 
    PressureConstraint pressure_constraint, const Indices& idx) {
    // full system and righthand-side including dirichlet boundary conditions
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_temp(dc.N_all, dc.N_all);
    
    setWeightsOseen(mat_temp, dc, idx);

    // reduced system without dirichlet boundary conditions
    eliminateDirichlet(mat, mat_temp, dc);

    if (pressure_constraint == PressureConstraint::SET && dc.idxs_neu.size() == 0)
    {
        // set last 2 rows and last column to zero and set 2x2 block to identity, removes pressure constraint
        // and sets pressure at node N_p to value of exact pressure solution at node N_p (N_p corresponds to N_dofs-2 overall)
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-3 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-2,dc.N_dofs-2) = 1.0;
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        mat.makeCompressed();
    }

    // don't need pressure constraint if Neumann boundary conditions are present
    if (dc.idxs_neu.size() > 0) {
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-2 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        mat.makeCompressed();
    }
}

OseenMatrix createMatrix(const OseenDiscretizationBetter& dc, PressureConstraint pressure_constraint, const Indices& idx)
{
    // full system and righthand-side including dirichlet boundary conditions
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_temp(dc.N_all, dc.N_all);
    
    OseenMatrix oseen_mat_with_dirbnd = setWeightsOseen(dc, idx);

    // reduced system without dirichlet boundary conditions
    OseenMatrix oseen_matrix = eliminateDirichlet(oseen_mat_with_dirbnd, dc);

    if (pressure_constraint == PressureConstraint::SET && dc.idxs_neu.size() == 0)
    {
        // set last 2 rows and last column to zero and set 2x2 block to identity, removes pressure constraint
        // and sets pressure at node N_p to value of exact pressure solution at node N_p (N_p corresponds to N_dofs-2 overall)
        for (int k=0; k<oseen_matrix.mat_schur.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(oseen_matrix.mat_schur,k); it; ++it)
            {
                if (it.row()>dc.N_p-2 || it.col()>dc.N_p-1)
                    oseen_matrix.mat_schur.coeffRef(it.row(),it.col()) = 0.0;
            }
        oseen_matrix.mat_schur.coeffRef(dc.N_p-1,dc.N_p-1) = 1.0;
        oseen_matrix.mat_schur.coeffRef(dc.N_p,dc.N_p) = 1.0;
        oseen_matrix.mat_schur.makeCompressed();
    }

    // don't need pressure constraint if Neumann boundary conditions are present
    if (dc.idxs_neu.size() > 0) {
        for (int k=0; k<oseen_matrix.mat_schur.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(oseen_matrix.mat_schur,k); it; ++it)
            {
                if (it.row()>dc.N_p-1 || it.col()>dc.N_p-1)
                    oseen_matrix.mat_schur.coeffRef(it.row(),it.col()) = 0.0;
            }
        oseen_matrix.mat_schur.coeffRef(dc.N_p,dc.N_p) = 1.0;
        oseen_matrix.mat_schur.makeCompressed();
    }

    return oseen_matrix;
}

// create Matrix that contains only Gradient and divergence with Dirichlet boundary nodes eliminated
// this is useful for restricted stencil preconditioner to avoid recomputing the velocity matrix
void createMatrixGradDiv(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const OseenDiscretizationBetter& dc, 
    PressureConstraint pressure_constraint) {
    // full system and righthand-side including dirichlet boundary conditions
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_temp(dc.N_all, dc.N_all);
    
    setWeightsGradDiv(mat_temp, dc);

    // reduced system without dirichlet boundary conditions
    eliminateDirichletGradDiv(mat, mat_temp, dc);

    if (pressure_constraint == PressureConstraint::SET && dc.idxs_neu.size() == 0)
    {
        // set last 2 rows and last column to zero and set 2x2 block to identity, removes pressure constraint
        // and sets pressure at node N_p to value of exact pressure solution at node N_p (N_p corresponds to N_dofs-2 overall)
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-3 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-2,dc.N_dofs-2) = 1.0;
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        mat.makeCompressed();
    }

    // don't need pressure constraint if Neumann boundary conditions are present
    if (dc.idxs_neu.size() > 0) {
        for (int k=0; k<mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
            {
                if (it.row()>dc.N_dofs-2 || it.col()>dc.N_dofs-2)
                    mat.coeffRef(it.row(),it.col()) = 0.0;
            }
        mat.coeffRef(dc.N_dofs-1,dc.N_dofs-1) = 1.0;
        mat.makeCompressed();
    }
}

} // namespace mm

#endif
