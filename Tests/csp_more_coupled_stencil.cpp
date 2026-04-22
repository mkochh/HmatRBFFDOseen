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

// N is the upper bound (exclusive), idx contains indices in [0, N)
std::vector<int> getMissingIndices(const std::vector<int>& idx, int N, int offset) {
    std::vector<uint8_t> present(N, false);
    for (int i : idx) {
        present[i-offset] = true;
    }
    std::vector<int> missing;
    missing.reserve(N - idx.size());
    for (int i = 0; i < N; ++i) {
        if (!present[i])
            missing.push_back(i);
    }
    return missing;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> 
updateMat(const Eigen::SparseMatrix<double, Eigen::RowMajor>& source_old, 
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& source_new, const Range<int>& idx, int idx_offset)
{
    std::vector<int> idx_missing = getMissingIndices(idx, source_old.rows(), idx_offset);

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(source_new.nonZeros());

    for (int i : idx_missing)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(source_old,i); it; ++it) {
            tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }

    for (int i : idx)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(source_new,i-idx_offset); it; ++it) {
            tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }

    Eigen::SparseMatrix<double, Eigen::RowMajor> updated_mat(source_old.rows(), source_old.cols());
    updated_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    updated_mat.makeCompressed();

    return updated_mat;
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

OseenMatrix updateOseenMatrix(const OseenMatrix& source_old, const OseenMatrix& source_new, 
    const OseenDiscretizationBetter& dc, const Indices& idx)
{
    int N = dc.N_ui + dc.N_uneu;
    int N_ub = dc.N_ub;
    OseenMatrix oseen_prcd;

    mm::Range<int> idx_ghost_vel(idx.idxs_neu_vel.size());
    for (int i = 0; i < idx.idxs_neu_vel.size(); i++)
        idx_ghost_vel[i] = idx.idxs_neu_vel[i] + dc.N_ui; // ghost nodes are offset from neumann nodes by N_ui

    mm::Range<int> idx_ghost_grad(idx.idxs_neu_grad.size());
    for (int i = 0; i < idx.idxs_neu_grad.size(); i++)
        idx_ghost_grad[i] = idx.idxs_neu_grad[i] + dc.N_ui; // ghost nodes are offset from neumann nodes by N_ui

    oseen_prcd.mat_vel = updateMat(source_old.mat_vel, source_new.mat_vel, idx.idxs_vel + idx_ghost_vel, N_ub);
    for (int i = 0; i < 3; i++) {
        oseen_prcd.mat_div[i] = updateMat(source_old.mat_div[i], source_new.mat_div[i], idx.idxs_div, N + N_ub);
        oseen_prcd.mat_grad[i] = updateMat(source_old.mat_grad[i], source_new.mat_grad[i], idx.idxs_grad + idx_ghost_grad, N_ub);
    }
    oseen_prcd.mat_schur = source_old.mat_schur;

    return oseen_prcd;
}

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

Csp_Data getCSP(pspmatrix mat_h2, OseenMatrix& prcd_oseen, pcluster rootv, pcluster rootp, AdmissibilityOptions adm_opt) {
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
                    tm, 0.1, 0.1, 0.1};

    Csp_Data csp_data = compute_csp(rootv, rootp, opt);

    del_truncmode(tm);
    del_spmatrix(reordered_prcd);

    return csp_data;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> BlockMatrix2Monolithic(const OseenMatrix& oseen_matrix)
{
    int N_vel = oseen_matrix.mat_vel.rows();
    int N_p = oseen_matrix.mat_schur.rows();
    Eigen::SparseMatrix<double, Eigen::RowMajor> monolithic_mat(3*N_vel + N_p, 3*N_vel + N_p);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(oseen_matrix.mat_vel.nonZeros() + oseen_matrix.mat_grad[0].nonZeros() + 
        oseen_matrix.mat_div[0].nonZeros());

    // set velocity block
    for (int k=0; k<oseen_matrix.mat_vel.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(oseen_matrix.mat_vel,k); it; ++it)
            tripletList.emplace_back(it.row(), it.col(), it.value());

    // set divergence blocks
    for (int k=0; k<oseen_matrix.mat_div[0].outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(oseen_matrix.mat_div[0],k); it; ++it)
            tripletList.emplace_back(it.row() + 3*N_vel, it.col(), it.value());

    // set gradient blocks
    for (int k=0; k<oseen_matrix.mat_grad[0].outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(oseen_matrix.mat_grad[0],k); it; ++it)
            tripletList.emplace_back(it.row(), it.col() + 3*N_vel, it.value());

    monolithic_mat.setFromTriplets(tripletList.begin(), tripletList.end());

    monolithic_mat.makeCompressed();

    return monolithic_mat;
}

// solve oseen with HLU
void test_strategy(string file_basename, string domain_name, RBFFDOptions rbffd_opt, 
    ClusteringOptions cluster_opt, AdmissibilityOptions adm_opt, real heps, HArith harith, TruncType trunc_type, int cg_support_size)
{
    // set input file for polyhedron
    std::stringstream OFF_File_base;
    OFF_File_base << "/OFF_Files/" << domain_name << ".off";
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
    OseenMatrix oseen_matrix(dc.N_ui + dc.N_uneu, dc.N_p, mat);
    auto t_create_end = high_resolution_clock::now();

    auto t_clustering_start = high_resolution_clock::now();
    // cluster_opt.max_depth_near = std::max(static_cast<int>(1.85*std::log2(std::log2(0.0025*(dc.N_ui + dc.N_uneu)))), 0); // heuristic
    // std::cout << "max_depth: " << cluster_opt.max_depth_near << std::endl;
    ClusteringOptions cluster_opt_temp = cluster_opt;
    cluster_opt_temp.velocity_cluster_type = VelocityClusterType::STANDARD_DD;
    auto [rootv, rootp] = getClustering(mat, dc, cluster_opt_temp, cg_support_size);
    auto t_clustering_end = high_resolution_clock::now();

    auto t_prcd_setup_start = high_resolution_clock::now();
    pspmatrix mat_prcd_h2;
    // psparsematrix schur_sp_h2 = nullptr;
    OseenMatrix oseen_prcd;
    {
    uint max_depth = std::max(getdepth_cluster(rootv), getdepth_cluster(rootp))*cluster_opt.depth_factor; // max_depth=cluster_depth/2 is heuristic 
    std::cout << "max_depth: " << max_depth << std::endl;

    auto t_determine_update_support_start = high_resolution_clock::now();
    // Indices idx_to_be_updated = setSearchAmongForNodesOseen(dc, rootv, rootp->son[0], mat, cluster_opt.max_depth_near);
    // Indices idx_to_be_updated = dc.determineSupportsGraph(rootv, rootp->son[0], mat, -1);
    Indices idx_to_be_updated = dc.determineSupportsGreedy(rootv, rootp->son[0], mat, max_depth);
    auto t_determine_update_support_end = high_resolution_clock::now();

    auto t_create_update_start = high_resolution_clock::now();
    OseenMatrix oseen_matrix_update = createMatrix(dc, rbffd_opt.pressure_constraint, idx_to_be_updated);
    auto t_create_update_end = high_resolution_clock::now();

    auto t_update_start = high_resolution_clock::now();
    oseen_prcd = updateOseenMatrix(oseen_matrix, oseen_matrix_update, dc, idx_to_be_updated);
    auto t_update_end = high_resolution_clock::now();
    mat_prcd_h2 = convertOseen2H(oseen_prcd); // convert to H2Lib-Sparsematrix format
    auto t_convert_end = high_resolution_clock::now();

    duration<double> duration_determine_update_support = t_determine_update_support_end - t_determine_update_support_start;
    duration<double> duration_create_update = t_create_update_end - t_create_update_start;
    duration<double> duration_update = t_update_end - t_update_start;
    duration<double> duration_convert = t_convert_end - t_update_end;
    std::cout << "time to determine supp.   : " << duration_determine_update_support.count() << std::endl;
    std::cout << "time to create update     : " << duration_create_update.count() << std::endl;
    std::cout << "time to update            : " << duration_update.count() << std::endl;
    std::cout << "time to convert           : " << duration_convert.count() << std::endl;

    // psparsematrix mat1 = matMM2H(mat);
    // Eigen::SparseMatrix<double, Eigen::RowMajor> mat_prcd_temp = BlockMatrix2Monolithic(oseen_prcd);
    // psparsematrix mat2 = matMM2H(mat_prcd_temp);
    // prn(norm2diff_sparsematrix(mat1, mat2));
    }

    if (cluster_opt.velocity_cluster_type == VelocityClusterType::COUPLED_DD) {
        pcluster rootv_co_dd = getCO_DD_VelClustering(oseen_prcd.mat_vel, oseen_prcd.mat_div[0], oseen_prcd.mat_grad[0], dc, cluster_opt, rootp, cg_support_size);
        del_cluster(rootv);
        rootv = rootv_co_dd;
    }
    auto t_prcd_setup_end = high_resolution_clock::now();

    Csp_Data csp_data = getCSP(mat_prcd_h2, oseen_prcd, rootv, rootp, adm_opt);

    // post-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DiscretizationData disc_data{dc.N_ui, dc.N_uneu, dc.N_p, dc.N_dofs, dc.dx_u, dc.exact_solution};

    writeCSPdata2CSV(file_basename, disc_data, csp_data, mat.nonZeros());

    // clean up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del_spmatrix2(mat_prcd_h2);
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

    std::array<string, 2> domain_name = {"cube", "bunny"};
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
    // 0 - COUPLED_WITH_INTERFACE, 1 - COUPLED_NO_INTERFACE, 2 - UNCOUPLED_GEOM
    if (argc > i)
        cluster_opt.pressure_cluster_type = static_cast<PressureClusterType>(atoi(argv[i]));
    i++;
    if (argc > i)
        cluster_opt.depth_factor = atof(argv[i]);
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
    if (argc > i)
        cluster_opt.connectivity_degree = atoi(argv[i]);
    i++;
    AdmissibilityType sp_adm_type = AdmissibilityType::SPARSE_ROW_COL;
    if (argc > i) {
        sp_adm_type = static_cast<AdmissibilityType>(atoi(argv[i]));
        if (sp_adm_type == AdmissibilityType::SPARSE_NNZ) {
            adm_opt.adm_vel = admissible_sparse;
            adm_opt.adm_grad = admissible_sparse;
            adm_opt.adm_div = admissible_sparse;
        } else if (sp_adm_type == AdmissibilityType::SPARSE_ROW_COL) {
            adm_opt.adm_vel = admissible_sparse_row_col;
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
    auto p = file_base.precision();
    file_base.precision(2);
    file_base << "Daten/csp_" << domain_name[static_cast<int>(rbffd_opt.domain_geometry)] << "_res_sten_depth" << cluster_opt.depth_factor;
    file_base.setf(std::ios::scientific);
    file_base.precision(0);
    file_base << "_" << partition_name[static_cast<int>(cluster_opt.partition_type)] << "_" << separator_name[static_cast<int>(cluster_opt.separator_type)] 
    << "_" << velocity_cluster_name[static_cast<int>(cluster_opt.velocity_cluster_type)] << "_" << pressure_cluster_name[static_cast<int>(cluster_opt.pressure_cluster_type)]
    << "_mlv_" << cluster_opt.max_leaf_size_vel << "_mlp_" << cluster_opt.max_leaf_size_p << "_msad_" << cluster_opt.min_size_after_disection
    << "_cd_" << cluster_opt.connectivity_degree
    << "_" << admissibility_name[static_cast<int>(sp_adm_type)] << adm_opt.nmax_vel;
    file_base.unsetf(std::ios::scientific);
    file_base.precision(p);
    file_base << admissibility_name[static_cast<int>(adm_opt.schur_adm_type)];
    if (adm_opt.schur_adm_type == AdmissibilityType::STRONG)
        file_base << adm_opt.eta_schur;
    else if (adm_opt.schur_adm_type == AdmissibilityType::SPARSE_NNZ || adm_opt.schur_adm_type == AdmissibilityType::SPARSE_ROW_COL)
        file_base << adm_opt.nmax_schur;
    file_base << "_cgss_" << cg_support_size;
    file_base.setf(std::ios::scientific);
    file_base.precision(0);
    file_base << "_lcgd_" << rbffd_opt.poly_lap <<  rbffd_opt.poly_conv << rbffd_opt.poly_grad  << rbffd_opt.poly_div << "_seed_" << rbffd_opt.seed;
    file_base.unsetf(std::ios::scientific);
    file_base.precision(p);
    file_base << "_s_" << rbffd_opt.step_size_scale << "_subset_" << static_cast<int>(rbffd_opt.subset)
    << "_neum_" << static_cast<int>(rbffd_opt.neumann)
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
        dx_min = 0.0032;
    for (; rbffd_opt.dx_u > dx_min; rbffd_opt.dx_u/=1.2)
        test_strategy(file_base.str(), domain_name[static_cast<int>(rbffd_opt.domain_geometry)], 
            rbffd_opt, cluster_opt, adm_opt, heps, harith, trunc_type, cg_support_size);

    uninit_h2lib();

    return 0;
}
