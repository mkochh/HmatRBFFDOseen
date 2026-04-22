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

// solve oseen with ILUT
void test_strategy(string file_basename, string domain_name, RBFFDOptions rbffd_opt, double droptol, int fillfactor)
{
    // set input file for polyhedron
    std::stringstream OFF_File_base;
    OFF_File_base << "OFF_Files/" << domain_name << ".off";
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

    // compute clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_clustering_start = high_resolution_clock::now();
    // auto [rootv, rootp] = getClustering(mat, dc, cluster_opt);
    // std::vector<int> perm(dc.N_dofs);
    // int N_offset = dc.N_ui + dc.N_uneu;
    // for (int i = 0; i < N_offset; i++) {
    //     perm[i] = rootv->idx[i];
    //     perm[i + N_offset] = rootv->idx[i] + N_offset;
    //     perm[i + 2*N_offset] = rootv->idx[i] + 2*N_offset;
    // }
    // for (int i = 0; i < dc.N_p+1; i++) {
    //     perm[3*N_offset + i] = rootp->idx[i] + 3*N_offset;
    // }
    auto t_clustering_end = high_resolution_clock::now();

    // Set up preconditioner and store hlu_times %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t_prcd_start = high_resolution_clock::now();
    Eigen::IncompleteLUT<double> ilut(mat, droptol, fillfactor); // uses AMD reordering
    // ILUTCustomPerm<double> ilut(mat, perm, droptol, fillfactor);
    auto t_prcd_end = high_resolution_clock::now();

    // std::ofstream file("Daten/idx.m");
    // file << "p = [";
    // for (int i = 0; i < perm.size() - 1; i++)
    //     file << perm[i] << ", ";
    // file << perm[perm.size()-1] << "];";
    // file.close();
    // writeMatrix2File(mat);
    // writeMatrix2File(ilut.getLU(),"ilut");
    
    // Solve problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    real tol_bicgstab = 1e-6; // bicgstab solver relative tolerance
    int iter = 500; // bicgstab maximal number of iterations
    Eigen::VectorXd sol = Eigen::VectorXd::Zero(mat.cols());

    auto t_solve_start = high_resolution_clock::now();
    bicgstab(mat, rhs, sol, ilut, iter, tol_bicgstab);
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
    duration<double> duration_prcd = t_prcd_end - t_prcd_start;    
    times.time_prcd = duration_prcd.count() + times.time_prcd_setup+ times.time_clustering;
    duration<double> duration_solve = t_solve_end - t_solve_start;
    times.time_solve = duration_solve.count();
    times.time_prcd_solve = times.time_prcd + times.time_solve;
    times.time_total = times.time_total_rbffd + times.time_prcd_solve;

    std::cout << "Time prcd:  " << times.time_prcd << std::endl;
    std::cout << "Time solve: " << times.time_solve << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Residual:   " << (mat * sol - rhs).norm() / rhs.norm() << std::endl;

    // DiscretizationData disc_data{dc.N_ui, dc.N_uneu, dc.N_p, dc.N_dofs, dc.dx_u, dc.exact_solution};

    // writeData2CSV(file_basename, disc_data, rhs, mat, sol_h2, mat_h2, P.get(), iter,
    //     hlu_times, times, hlu_memory, hlu_stats, tol_bicgstab);
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

    std::array<string, 3> domain_name = {"cube", "bunny", "long_cuboid"};

    std::array<string, 3> pressure_constraint_name = {"POLY_QUAD", "SET", "AVERAGE"};

    RBFFDOptions rbffd_opt;

    // truncation accuracy parameter for HLU
    double droptol = 0.1;
    if (argc > i)
        droptol = atof(argv[i]);
    i++;
    int fillfactor = 10;
    if (argc > i)
        fillfactor = atoi(argv[i]);
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
    << "_droptol_" << droptol << "_fillfac_" << fillfactor;
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
            rbffd_opt, droptol, fillfactor);

    uninit_h2lib();

    return 0;
}
