#include <medusa/Medusa_fwd.hpp>
#include <medusa/bits/domains/PolyhedronShape.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <Eigen/SparseCore>
#include <math.h>
#include <chrono>
#include <memory>
#include <array>
#include <algorithm>
#include <numeric>

#include "../h2libext.h"

using namespace mm;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

enum class AdmissibilityType {
    SPARSE_ROW_COL,
    SPARSE_NNZ,
    STRONG
};

struct AdmissibilityOptions {
    // admissibility condition options
    // maximum number of nonzeros per block or per maximum number of rows/columns per block
    // depending on admissibility condition (either admissible_sparse or admissible_dd_sparse_row_col)
    int nmax_vel = 20; 
    int nmax_grad = 20;
    int nmax_div = 20;
    real eta_schur = 32.0; // eta for strong admissibility condition for schur complement
    int nmax_schur = 20;

    AdmissibilityType schur_adm_type = AdmissibilityType::STRONG;

    admissible adm_vel = admissible_dd_sparse_row_col;
    admissible adm_grad = admissible_sparse_row_col;
    admissible adm_div = admissible_sparse_row_col;
    admissible adm_schur = admissible_2_min_cluster_rbffd;
};

enum class TruncType {
    RANDOM,
    LANCZOS,
    SVD
};

reordered_sparsematrix getReorderedSparseMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& sp, uint* rowinv, uint* colinv, bool transpose = false)
{
    psparsematrix rsp;
    uint *col, *row;           // Column / Row permutation

    col = new uint[sp.cols()];
    row = new uint[sp.rows()];

    for (uint i = 0; i < sp.cols(); i++)
        col[colinv[i]] = i;
    for (uint i = 0; i < sp.rows(); i++)
        row[rowinv[i]] = i;

    Eigen::PermutationMatrix<Eigen::Dynamic> P_row(sp.rows()), P_col(sp.cols());
    for (int i = 0; i < sp.rows(); i++)
        P_row.indices()[i] = row[i];

    for (int i = 0; i < sp.cols(); i++)
        P_col.indices()[i] = colinv[i];

    Eigen::SparseMatrix<double, Eigen::RowMajor> temp = P_row * sp;
    sp = temp * P_col;

    if (transpose) {
        rsp = matMM2H(sp.transpose());
        return {rsp, row, col};
    }
    
    rsp = matMM2H(sp);

    return {rsp, col, row};
}

void reorderSparseMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& sp, uint* rowinv, uint* colinv)
{
    std::vector<int> row(sp.rows());

    for (uint i = 0; i < sp.rows(); i++)
        row[rowinv[i]] = i;

    Eigen::PermutationMatrix<Eigen::Dynamic> P_row(sp.rows()), P_col(sp.cols());
    for (int i = 0; i < sp.rows(); i++)
        P_row.indices()[i] = row[i];

    for (int i = 0; i < sp.cols(); i++)
        P_col.indices()[i] = colinv[i];

    Eigen::SparseMatrix<double, Eigen::RowMajor> temp = P_row * sp;
    sp = temp * P_col;
}

pspmatrix convertOseen2H(const OseenMatrix& oseen_matrix)
{
    psparsematrix *B = (psparsematrix *)allocmem(6 * sizeof(psparsematrix));
    // set gradient [3,4,5] and divergence [0,1,2] blocks
    for (int i = 0; i < 3; i++)
        B[i] = matMM2H(oseen_matrix.mat_div[i]);

    for (int i = 0; i < 3; i++)
        B[i+3] = matMM2H(oseen_matrix.mat_grad[i].transpose());

    // set set lower right block
    psparsematrix C = matMM2H(oseen_matrix.mat_schur);

    // set velocity block
    psparsematrix A = matMM2H(oseen_matrix.mat_vel);

    pspmatrix mat_oseen_h2 = new_spmatrix(A, B, C); // stiffness matrix

    return mat_oseen_h2;
}

std::unique_ptr<Block_HLU_Prcd> createBlockHLUPrcd(pspmatrix mat_h2, OseenMatrix& prcd_oseen, double heps, Block_HLU_Times& hlu_times,
    HLU_Memory& hlu_memory, HLU_Stats& hlu_stats, pcluster rootv, pcluster rootp, AdmissibilityOptions adm_opt,
    HArith harith, TruncType trunc_type) {
    // Set up HLU_Options

    ptruncmode tm = new_releucl_truncmode(); // truncation mode for when standard H-arithmetic is used


    // reorders mat_vel, mat_div[0],... in-place and returns them in h2lib format
    auto [rA, colA, rowA] = getReorderedSparseMatrix(prcd_oseen.mat_vel, rootv->idx, rootv->idx);

    Eigen::SparseMatrix<double, Eigen::RowMajor> schur_sp;
    std::unique_ptr<adm_sparse_data> sparse_value_schur;
    void* eta_schur;
    bool schur_sp_adm = adm_opt.schur_adm_type == AdmissibilityType::SPARSE_NNZ || adm_opt.schur_adm_type == AdmissibilityType::SPARSE_ROW_COL;

    if (schur_sp_adm) {
        schur_sp = prcd_oseen.mat_div[0] * prcd_oseen.mat_grad[0];
        auto [rSchur, colSchur, rowSchur] = getReorderedSparseMatrix(schur_sp, rootp->idx, rootp->idx);
        sparse_value_schur = std::make_unique<adm_sparse_data>(adm_opt.nmax_schur, rSchur, colSchur, rowSchur);
        eta_schur = (void *)&sparse_value_schur;
    } else {
        eta_schur = (void *)&adm_opt.eta_schur;
    }
    
    auto [rDiv, colDiv, rowDiv] = getReorderedSparseMatrix(prcd_oseen.mat_div[0], rootp->idx, rootv->idx);
    auto [rGrad, colGrad, rowGrad] = getReorderedSparseMatrix(prcd_oseen.mat_grad[0], rootv->idx, rootp->idx, true);
    
    // any block with more than nmax non-zero rows or columns is inadmissible
    std::unique_ptr<adm_sparse_data> sparse_value_vel = std::make_unique<adm_sparse_data>(adm_opt.nmax_vel, rA, colA, rowA);
    std::unique_ptr<adm_sparse_data> sparse_value_grad = std::make_unique<adm_sparse_data>(adm_opt.nmax_grad, rGrad, colGrad, rowGrad);
    std::unique_ptr<adm_sparse_data> sparse_value_div = std::make_unique<adm_sparse_data>(adm_opt.nmax_div, rDiv, colDiv, rowDiv);
    
    void *eta_vel, *eta_grad, *eta_div; // etas for admissibility condtions
    eta_vel = (void *)&sparse_value_vel;
    eta_grad = (void *)&sparse_value_grad;
    eta_div = (void *)&sparse_value_div;

    // reorders matrices in-place that were not previously reordered
    reorderSparseMatrix(prcd_oseen.mat_div[1], rootp->idx, rootv->idx);
    reorderSparseMatrix(prcd_oseen.mat_div[2], rootp->idx, rootv->idx);
    reorderSparseMatrix(prcd_oseen.mat_grad[1], rootv->idx, rootp->idx);
    reorderSparseMatrix(prcd_oseen.mat_grad[2], rootv->idx, rootp->idx);
    reorderSparseMatrix(prcd_oseen.mat_schur, rootp->idx, rootp->idx);

    pspmatrix reordered_prcd = convertOseen2H(prcd_oseen);

    HLU_Options opt{reordered_prcd, eta_vel, eta_grad, eta_div, eta_schur,
                    adm_opt.adm_vel, adm_opt.adm_grad, adm_opt.adm_div, adm_opt.adm_schur, 
                    tm, heps, heps, heps};
    
    std::unique_ptr<TruncationOperator> trunc; // truncation operator for when SumExpression H-arithmetic is used
    switch(trunc_type) {
        case TruncType::RANDOM:
            trunc = std::make_unique<RandomTruncation>(opt.eps_vel, 1);
            break;
        case TruncType::LANCZOS:
            trunc = std::make_unique<LanczosTruncation>(opt.eps_vel, 1);
            break;
        case TruncType::SVD:
            trunc = std::make_unique<SVDTruncation>(opt.eps_vel);
            break;
    }
    std::unique_ptr<Block_HLU_Prcd> P = std::make_unique<Block_HLU_Prcd>(rootv, rootp, opt, hlu_times, hlu_memory, hlu_stats, BLOCK_TRIANGULAR, 
        harith, trunc);

    // need the unordered B and C matrices (grad, div, schur) later in apply_preconditioner in bicgstab routine
    // schur/C technically not needed, but included for consistency
    P->B = mat_h2->B;
    P->C = mat_h2->C;

    del_truncmode(tm);
    del_spmatrix(reordered_prcd);

    return P;
}

std::vector<double> test_separation_distance(const mm::DomainDiscretization<mm::Vec3d>& d_const)
{
    mm::DomainDiscretization<mm::Vec3d> d = d_const;
    FindClosest find_closest(2);
    find_closest.forceSelf(true);
    d.findSupport(find_closest);
    std::vector<double> sep_dist(d.size());
    for (int i = 0; i < d.size(); i++) {
        sep_dist[i] = (d.pos(i) - d.supportNode(i,1)).norm();
    }
    std::vector<double>::iterator min = min_element(sep_dist.begin(), sep_dist.end());
    std::vector<double>::iterator max = max_element(sep_dist.begin(), sep_dist.end());
    double average = reduce(sep_dist.begin(), sep_dist.end())/sep_dist.size();
    std::cout << "min separation dist = " << *min << std::endl;
    std::cout << "max separation dist = " << *max << std::endl;
    std::cout << "average separation dist = " << average << std::endl;

    return sep_dist;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> getSchurSparsity(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    const OseenDiscretizationBetter& dc)
{
    int N = dc.N_ui + dc.N_uneu;
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_div = mat.block(3*N, 0, dc.N_p+1, N);
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_grad = mat.block(0, 3*N, N, dc.N_p+1);

    Eigen::SparseMatrix<double, Eigen::RowMajor> schur_sparsity = mat_div * mat_grad;
    // Eigen::SparseMatrix<double, Eigen::RowMajor> schur_sparsity_2 = schur_sparsity * schur_sparsity;

    // writeMatrix2File(schur_sparsity, "sp1");
    // writeMatrix2File(schur_sparsity_2, "sp2");

    return schur_sparsity;
    // return schur_sparsity_2;
}

class PrcdMatMult
{
    public:
        using Scalar = double;
        PrcdMatMult(const Eigen::SparseMatrix<double, Eigen::RowMajor>& m, Block_HLU_Prcd* prcd) : mat(m), P(prcd) {}
        int rows() const { return mat.rows(); }
        int cols() const { return mat.cols(); }

        void perform_op(const Scalar* x, Scalar* y) const {
            pavector x_h2 = new_avector(rows());
            for (int i = 0; i < rows(); i++)
                x_h2->v[i] = x[i];
            P->apply_preconditioner(x_h2);
            Eigen::Map<const Eigen::VectorXd> x_vec(x_h2->v, cols());
            Eigen::Map<Eigen::VectorXd> y_vec(y, rows());
            y_vec = mat * x_vec;
        }

    private:
        Eigen::SparseMatrix<double, Eigen::RowMajor> mat;
        Block_HLU_Prcd* P;
};

// solve oseen with HLU
void test_strategy(string file_basename, string domain_name, RBFFDOptions rbffd_opt, 
    ClusteringOptions cluster_opt, AdmissibilityOptions adm_opt, real heps, HArith harith, TruncType trunc_type, int cg_support_size)
{
    // set input file for polyhedron
    std::stringstream OFF_File_base;
    OFF_File_base << "/home/michael/Dokumente/Programming/TEST/OseenRBFFDH2Lib/Tests/OFF_Files/" << domain_name << ".off";
    string OFF_file = OFF_File_base.str();
    PolyhedronShape<Vec3d> shape = PolyhedronShape<Vec3d>::fromOFF(OFF_file);

    auto t_disc_start =  high_resolution_clock::now();
    // construct domains needed for discretization of Oseen equations and initialize parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    OseenDiscretizationBetter dc(shape, rbffd_opt); 
    auto t_disc_end = high_resolution_clock::now();
    
    // determine pressure constraint %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_con_start = high_resolution_clock::now();
    if (rbffd_opt.pressure_constraint == PressureConstraint::POLY_QUAD && dc.idxs_neu.size() == 0)
        dc.setConstraint(OFF_file); // set pressure constraint, needed to make compute pressure at high accuracy, if there are no neumann boundary nodes
    auto t_con_end = high_resolution_clock::now();

    // determine supports (stencils) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_find_supp_start = high_resolution_clock::now();
    dc.determineSupports();
    // dc.determineSupportsGraph();
    if (dc.use_hyperviscosity)
        dc.determineSupportsHyperViscosity();
    auto t_find_supp_end = high_resolution_clock::now();

    std::cout << "N_ui = " << dc.N_ui << " and N_p = " << dc.N_p << std::endl;
    std::cout << "N_dir = " << dc.N_ub << " and N_neu = " << dc.N_uneu << std::endl;

    // compute weights and remove dirichlet boundary conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_create_start = high_resolution_clock::now();
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat(dc.N_dofs,dc.N_dofs);
    Eigen::VectorXd rhs(dc.N_dofs);
    createMatrixAndRHS(mat, rhs, dc, rbffd_opt.pressure_constraint);
    auto t_create_end = high_resolution_clock::now();

    // Eigen::SparseMatrix<double, Eigen::RowMajor> mat_vel = mat.block(0, 0, dc.N_ui + dc.N_uneu, dc.N_ui + dc.N_uneu);
    // writeMatrix2File(mat_vel, "conv_diff");

    // compute clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_clustering_start = high_resolution_clock::now();
    auto [rootv, rootp] = getClustering(mat, dc, cluster_opt, cg_support_size);
    // auto [rootv, rootp] = getBlackboxClustering(mat, dc.N_ui+dc.N_uneu, dc.N_p, dc.d_u.dim, cluster_opt);
    auto t_clustering_end = high_resolution_clock::now();

    if (rootv->sons == 3)
        prn(rootv->son[2]->size);
    if (rootv->son[0]->sons == 3)
        prn(rootv->son[0]->son[2]->size);
    if (rootv->son[1]->sons == 3)
        prn(rootv->son[1]->son[2]->size);

    // H2Lib conversions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    OseenMatrix oseen_mat(dc.N_ui + dc.N_uneu, dc.N_p, mat);
    pspmatrix mat_h2 = convertOseen2H(oseen_mat); // convert to H2Lib-Sparsematrix format
    pavector rhs_h2 = vecMM2H(rhs); // rhs after eliminating boundary conditions in H2Lib-format
    pavector sol_h2 = new_zero_avector(dc.N_dofs); // vector for numeric solution in H2Lib-format

    // Set up preconditioner and store hlu_times %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_prcd_start = high_resolution_clock::now();
    Block_HLU_Times hlu_times;
    HLU_Memory hlu_memory;
    HLU_Stats hlu_stats;
    std::unique_ptr<Block_HLU_Prcd> P = createBlockHLUPrcd(mat_h2, oseen_mat, heps, hlu_times, hlu_memory, hlu_stats, rootv, rootp, 
             adm_opt, harith, trunc_type);
    auto t_prcd_end = high_resolution_clock::now();

    // PrcdMatMult prcd_mult_op(mat, P.get());
    // Spectra::GenEigsSolver<PrcdMatMult> eigs(prcd_mult_op, 3, 6);
    // eigs.init();
    // eigs.compute(Spectra::SortRule::LargestMagn);
    // if(eigs.info() == Spectra::CompInfo::Successful)
    // {
    //     Eigen::VectorXcd evalues = eigs.eigenvalues();
    //     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    // } else {
    //     std::cout << "Eigenvalue computation did not converge." << std::endl;
    // }
    // eigs.compute(Spectra::SortRule::SmallestReal);
    // if(eigs.info() == Spectra::CompInfo::Successful)
    // {
    //     Eigen::VectorXcd evalues = eigs.eigenvalues();
    //     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    // } else {
    //     std::cout << "Eigenvalue computation did not converge." << std::endl;
    // }
    // eigs.compute(Spectra::SortRule::SmallestMagn);
    // if(eigs.info() == Spectra::CompInfo::Successful)
    // {
    //     Eigen::VectorXcd evalues = eigs.eigenvalues();
    //     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    // } else {
    //     std::cout << "Eigenvalue computation did not converge." << std::endl;
    // }
    
    // Solve problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    real tol_bicgstab = 1e-6; // bicgstab solver relative tolerance
    uint maxiter = 500; // bicgstab maximal number of iterations
    BiCGStab bicgstab(tol_bicgstab, maxiter);
    // GMRes gmres(tol_bicgstab, maxiter, 100);

    auto t_solve_start = high_resolution_clock::now();
    uint iter = bicgstab.psolve_avector_eigen(mat_h2, (addeval_t)addeval_spmatrix_avector_rbffd,
                                        P.get(), rhs_h2, sol_h2);
    // uint iter = gmres.psolve_avector(mat_h2, (addeval_t)addeval_spmatrix_avector_rbffd,
    //                                     P.get(), rhs_h2, sol_h2);
    auto t_solve_end = high_resolution_clock::now();

    // post-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Timings times;
    duration<double> duration_disc = t_disc_end - t_disc_start;
    times.time_disc = duration_disc.count();
    duration<double> duration_con = t_con_end - t_con_start;
    times.time_con = duration_con.count();
    duration<double> duration_supp = t_find_supp_end - t_find_supp_start;
    times.time_supp = duration_supp.count();
    duration<double> duration_assemble = t_create_end - t_create_start;
    times.time_create = duration_assemble.count();
    times.time_total_rbffd = times.time_disc + times.time_con + times.time_supp + times.time_create;
    duration<double> duration_clustering = t_clustering_end - t_clustering_start;    
    times.time_clustering = duration_clustering.count();
    times.time_prcd_setup = 0.0;
    duration<double> duration_prcd = t_prcd_end - t_prcd_start;    
    times.time_prcd = duration_prcd.count() + times.time_prcd_setup+ times.time_clustering;
    duration<double> duration_solve = t_solve_end - t_solve_start;
    times.time_solve = duration_solve.count();
    times.time_prcd_solve = times.time_prcd + times.time_solve;
    times.time_total = times.time_total_rbffd + times.time_prcd_solve;

    DiscretizationData disc_data{dc.N_ui, dc.N_uneu, dc.N_p, dc.N_dofs, dc.dx_u, dc.exact_solution};

    // psparsematrix mat_h2_mono = matMM2H(mat);
    // prn(norm2diff_id_tria_prcd(P.get(), mat_h2_mono));

    writeData2CSV(file_basename, disc_data, rhs, mat, sol_h2, mat_h2, P.get(), iter,
        hlu_times, times, hlu_memory, hlu_stats, tol_bicgstab);

    // clean up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del_avector(sol_h2);
    del_avector(rhs_h2);
    del_spmatrix2(mat_h2);
}

int main(int argc, char **argv) {
    int i = 1;
    #ifdef USE_OPENMP
    int threads = 8; // number of threads
    if (argc > i)
        threads = atoi(argv[i]);
    i++;
    omp_set_num_threads(threads);
    std::cout << "Using OpenMP with " << threads << " threads" << std::endl;
    #else
    std::cout << "Using serial implementation" << std::endl;
    #endif

    init_h2lib(&argc, &argv);

    std::array<string, 3> domain_name = {"cube", "bunny", "long_cuboid"};
    std::array<string, 2> partition_name = {"METIS", "GEOM"};
    std::array<string, 2> separator_name = {"SIMPLE", "MIN"};

    std::array<string, 2> velocity_cluster_name = {"ST_DD", "CO_DD"};
    std::array<string, 5> pressure_cluster_name = {"CO_INTER", "CO_NO_INTER", "UNCO_GEOM", "CO_1ZERO", "UNCO_METIS"};

    std::array<string, 3> admissibility_name = {"SP_ROW_COL", "SP_NNZ", "STRONG"};

    std::array<string, 2> harith_name = {"STD", "SUMEXP"};
    std::array<string, 3> trunc_name = {"RANDOM", "LANCZOS", "SVD"};

    std::array<string, 3> pressure_constraint_name = {"POLY_QUAD", "SET", "AVERAGE"};

    // standard options defined in definition of structs
    ClusteringOptions cluster_opt;
    AdmissibilityOptions adm_opt;
    RBFFDOptions rbffd_opt;

    // truncation accuracy parameter for HLU
    double heps = 0.1;
    if (argc > i)
        heps = atof(argv[i]);
    i++;
    // 0 - METIS, 1 - GEOM
    if (argc > i)
        cluster_opt.partition_type = static_cast<PartitionType>(atoi(argv[i]));
    i++;
    // 0 - SIMPLE, 1 - MINIMUM
    if (argc > i)
        cluster_opt.separator_type = static_cast<SeparatorType>(atoi(argv[i]));
    i++;
    // 0 - STANDAR_DD, 1 - COUPLED_DD
    if (argc > i)
        cluster_opt.velocity_cluster_type = static_cast<VelocityClusterType>(atoi(argv[i]));
    i++;
    // 0 - COUPLED_WITH_INTERFACE, 1 - COUPLED_NO_INTERFACE, 2 - UNCOUPLED_GEOM, 3 - COUPLED_ONE_ZERO_BLOCK, 4 - UNCOUPLED_METIS
    // if COUPLED_DD was chosen before, only UNCOUPLED_GEOM and UNCOUPLED_METIS are valid choices here
    if (argc > i)
        cluster_opt.pressure_cluster_type = static_cast<PressureClusterType>(atoi(argv[i]));
    i++;
    if (argc > i)
        cluster_opt.max_leaf_size_vel = atoi(argv[i]);
    i++;
    if (argc > i)
        cluster_opt.max_leaf_size_p = atoi(argv[i]);
    i++;  
    if (argc > i)
        cluster_opt.min_size_after_disection = atoi(argv[i]);
    i++;
    AdmissibilityType sp_adm_type = AdmissibilityType::SPARSE_ROW_COL;
    if (argc > i) {
        sp_adm_type = static_cast<AdmissibilityType>(atoi(argv[i]));
        if (sp_adm_type == AdmissibilityType::SPARSE_NNZ) {
            adm_opt.adm_vel = admissible_dd_sparse;
            adm_opt.adm_grad = admissible_sparse;
            adm_opt.adm_div = admissible_sparse;
        } else if (sp_adm_type == AdmissibilityType::SPARSE_ROW_COL) {
            adm_opt.adm_vel = admissible_dd_sparse_row_col;
            adm_opt.adm_grad = admissible_sparse_row_col;
            adm_opt.adm_div = admissible_sparse_row_col;
        }
    }
    i++;
    if (argc > i) {
        adm_opt.nmax_vel = atoi(argv[i]);
        adm_opt.nmax_grad = atoi(argv[i]);
        adm_opt.nmax_div = atoi(argv[i]);
    }
    i++;
    // 0 - SPARSE_ROW_COL, 1 - SPARSE_NNZ, 2 - STRONG
    if (argc > i) {
        adm_opt.schur_adm_type = static_cast<AdmissibilityType>(atoi(argv[i]));
        switch (adm_opt.schur_adm_type)
        {
        case AdmissibilityType::SPARSE_ROW_COL:
            adm_opt.adm_schur = admissible_sparse_row_col;
            break;
        case AdmissibilityType::SPARSE_NNZ:
            adm_opt.adm_schur = admissible_sparse;
            break;
        case AdmissibilityType::STRONG:
            adm_opt.adm_schur = admissible_2_min_cluster_rbffd;
            break;
        default:
            break;
        }
    }
    i++;
    // depending on previous choice, set either eta for strong adm. or nmax for sparse adm.
    if (argc > i) {
        if (adm_opt.schur_adm_type == AdmissibilityType::STRONG)
            adm_opt.eta_schur = atof(argv[i]);
        else if (adm_opt.schur_adm_type == AdmissibilityType::SPARSE_NNZ || adm_opt.schur_adm_type == AdmissibilityType::SPARSE_ROW_COL)
            adm_opt.nmax_schur = atoi(argv[i]);
    }
    i++;
    int cg_support_size = 0; // only center point is used in cluster geometry -> effects strong admissibility condition 
    if (argc > i)
        cg_support_size = atoi(argv[i]);
    i++;
    // choose arithmetic: 0 - STANDARD, 1 - SUMEXP
    HArith harith = HArith::STANDARD;
    if (argc > i)
        harith = static_cast<HArith>(atoi(argv[i]));
    i++;
    // choose truncation type: 0 - RANDOM, 1 - LANCZOS, 2 - SVD
    TruncType trunc_type = TruncType::RANDOM;
    if (argc > i)
        trunc_type = static_cast<TruncType>(atoi(argv[i]));
    i++;
    // choose domain: 0 - cube, 1 - bunny
    if (argc > i)
        rbffd_opt.domain_geometry = static_cast<DomainGeometry>(atoi(argv[i]));
    i++;
    // separation distance for velocity nodes
    if (argc > i)
        rbffd_opt.dx_u = atof(argv[i]);
    i++;
    // separation distance for pressure nodes = step_size_scale * separation distance for velocity nodes (dx_u)
    if (argc > i)
        rbffd_opt.step_size_scale = atof(argv[i]);
    i++;
    // determines whether pressure nodes are a subset of velocity nodes
    if (argc > i)
        rbffd_opt.subset = static_cast<bool>(atoi(argv[i]));
    i++;
    // random seed for point generation
    if (argc > i)
        rbffd_opt.seed = atoi(argv[i]);
    i++;
    // polynomial augmentation degree for laplace, convection, gradient and divergence
    // only degrees 1 to 5 are supported but this can be extended in OseenDiscretizationBetter
    if (argc > i)
        rbffd_opt.poly_lap = atoi(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.poly_conv = atoi(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.poly_grad = atoi(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.poly_div = atoi(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.pressure_constraint = static_cast<PressureConstraint>(atoi(argv[i]));
    i++;
    if (argc > i)
        rbffd_opt.use_hyperviscosity = static_cast<bool>(atoi(argv[i]));
    i++;
    if (argc > i)
        rbffd_opt.nu = atof(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.conv = atoi(argv[i]);
    i++; 
    if (argc > i)
        rbffd_opt.sol = atoi(argv[i]);
    i++;
    if (argc > i)
        rbffd_opt.neumann = static_cast<bool>(atoi(argv[i]));
    i++;
    // set file to write data to
    std::stringstream file_base;
    file_base.setf(std::ios::scientific);
    auto p = file_base.precision();
    file_base.precision(0);
    file_base << "Daten/" << domain_name[static_cast<int>(rbffd_opt.domain_geometry)] 
    << "_heps_" << heps << "_" << partition_name[static_cast<int>(cluster_opt.partition_type)] << "_" << separator_name[static_cast<int>(cluster_opt.separator_type)] 
    << "_" << velocity_cluster_name[static_cast<int>(cluster_opt.velocity_cluster_type)] << "_" << pressure_cluster_name[static_cast<int>(cluster_opt.pressure_cluster_type)]
    << "_mlv_" << cluster_opt.max_leaf_size_vel << "_mlp_" << cluster_opt.max_leaf_size_p << "_msad_" << cluster_opt.min_size_after_disection
    << "_" << admissibility_name[static_cast<int>(sp_adm_type)] << adm_opt.nmax_vel;
    file_base.unsetf(std::ios::scientific);
    file_base.precision(p);
    file_base << admissibility_name[static_cast<int>(adm_opt.schur_adm_type)];
    if (adm_opt.schur_adm_type == AdmissibilityType::STRONG)
        file_base << adm_opt.eta_schur;
    else if (adm_opt.schur_adm_type == AdmissibilityType::SPARSE_NNZ || adm_opt.schur_adm_type == AdmissibilityType::SPARSE_ROW_COL)
        file_base << adm_opt.nmax_schur;
    file_base << "_cgss_" << cg_support_size << "_";
    file_base << harith_name[static_cast<int>(harith)] << "_" << trunc_name[static_cast<int>(trunc_type)];
    file_base.setf(std::ios::scientific);
    file_base.precision(0);
    file_base << "_lcgd_" << rbffd_opt.poly_lap <<  rbffd_opt.poly_conv << rbffd_opt.poly_grad  << rbffd_opt.poly_div << "_nu_" << rbffd_opt.nu << "_seed_" << rbffd_opt.seed;
    file_base.unsetf(std::ios::scientific);
    file_base.precision(p);
    file_base << "_s_" << rbffd_opt.step_size_scale << "_subset_" << static_cast<int>(rbffd_opt.subset)
    << "_conv_" << rbffd_opt.conv << "_sol_" << rbffd_opt.sol << "_neum_" << static_cast<int>(rbffd_opt.neumann)
    << "_pres_" << pressure_constraint_name[static_cast<int>(rbffd_opt.pressure_constraint)] 
    << "_hyp_" << static_cast<int>(rbffd_opt.use_hyperviscosity);
    #ifdef USE_OPENMP
    file_base << "_omp_" << threads;
    #else
    file_base << "_serial";
    #endif

    // compute for different separation distances
    double dx_min = 1.0/120.0;
    if (rbffd_opt.domain_geometry == DomainGeometry::BUNNY)
        dx_min = 1.0/150.0;
    for (; rbffd_opt.dx_u > dx_min; rbffd_opt.dx_u/=1.2)
        test_strategy(file_base.str(), domain_name[static_cast<int>(rbffd_opt.domain_geometry)], 
            rbffd_opt, cluster_opt, adm_opt, heps, harith, trunc_type, cg_support_size);

    uninit_h2lib();

    return 0;
}
