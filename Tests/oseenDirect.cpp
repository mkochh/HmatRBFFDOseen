#include <medusa/Medusa_fwd.hpp>
#include <medusa/bits/domains/PolyhedronShape.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <Eigen/SparseCore>
#ifdef USE_UMFPACK
#include <Eigen/UmfPackSupport>
#else
#include <Eigen/SparseLU>
#endif
#include <math.h>
#include <chrono>

#include "../h2libext.h"

using namespace mm;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void print_header_to_file(string path)
{
    std::ofstream file(path);

    file << "poly_grad;";
    file << "poly_div;";
    file << "poly_conv;";
    file << "poly_lap;";
    file << "nu;";
    file << "N_ui;";
    file << "N_p;";
    file << "N_total;";
    file << "t_solve;";
    file << "solution_error;";
    file << "solution_error_u;";
    file << "solution_error_p;";
    file << "solution_error_inf;";
    file << "solution_error_u_inf;";
    file << "solution_error_p_inf;";
    file << "residual_error;";

    file << std::endl;

    file.close();
}

void writeData2CSV(string file_basename,
    const OseenDiscretizationBetter& dc, const Eigen::VectorXd& rhs, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, 
    const double time_solve, const Eigen::VectorXd& sol_vec) {
    // Compute distance to e
    Eigen::VectorXd u_exact = dc.exact_solution;
    int N = dc.N_ui + dc.N_uneu;
    int N_p = dc.N_p;
    double norm_sol = (sol_vec-u_exact).norm()/u_exact.norm();
    double norm_sol_inf = (sol_vec-u_exact).lpNorm<Eigen::Infinity>();
    double norm_sol_u = (sol_vec.head(3*N)-u_exact.head(3*N)).norm()/u_exact.head(3*N).norm();
    double norm_sol_u_inf = (sol_vec.head(3*N)-u_exact.head(3*N)).lpNorm<Eigen::Infinity>();
    double norm_sol_p = (sol_vec.segment(3*N,N_p)-u_exact.segment(3*N,N_p)).norm()/u_exact.segment(3*N,N_p).norm();
    double norm_sol_p_inf = (sol_vec.segment(3*N,N_p)-u_exact.segment(3*N,N_p)).lpNorm<Eigen::Infinity>();
    double norm_res = (rhs-mat*sol_vec).norm()/rhs.norm();

    // Print data to file
    std::stringstream outfile_name;
    outfile_name << file_basename << ".csv";
    // if file doesn't exist yet write header
    if (!std::ifstream(outfile_name.str()))
        print_header_to_file(outfile_name.str());

    std::ofstream outfile(outfile_name.str(), std::ofstream::app);
    auto default_precision = outfile.precision();

    outfile << dc.poly_grad << ";" << dc.poly_div << ";" << dc.poly_conv << ";" << dc.poly_lap << ";" << dc.nu
    << ";" << N << ";" << N_p << ";" << dc.N_dofs << ";";

    outfile.setf(std::ios::fixed);
    outfile.precision(2);
    outfile << time_solve << ";";
    outfile.unsetf(std::ios::fixed);
    outfile.precision(default_precision);

    outfile.setf(std::ios::scientific);
    outfile << norm_sol << ";" << norm_sol_u << ";" << norm_sol_p << ";" 
    << norm_sol_inf << ";" << norm_sol_u_inf << ";" << norm_sol_p_inf << ";"
    << norm_res << std::endl;
    outfile.unsetf(std::ios::scientific);

    outfile.close();
}

// solve oseen directly
void test_strategy(string file_basename, string domain_name, RBFFDOptions rbffd_opt)
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

    Eigen::SparseMatrix<double, Eigen::ColMajor> mat_col = mat;
    
    // Solve problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t6 = high_resolution_clock::now();
    // for UmfPack, from user guide:
    // if the analysis requires more than 2GB of memory and you are using
    // the int32_t version of UMFPACK, then you are guaranteed
    // to run out of memory. Try using the 64-bit version of UMFPACK.
    // 64-bit version of UMFPACK requires 64-bit BLAS
    #ifdef USE_UMFPACK
    Eigen::UmfPackLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> solver;
    #else
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    #endif
    solver.compute(mat_col);
    Eigen::VectorXd sol_vec = solver.solve(rhs);
    auto t7 = high_resolution_clock::now();

    #ifdef USE_UMFPACK
    solver.printUmfpackInfo();
    solver.printUmfpackStatus();
    #endif

    // post-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    duration<double> duration_solve = t7 - t6;
    double time_solve = duration_solve.count();

    writeData2CSV(file_basename, dc, rhs, mat, time_solve, sol_vec);
    std::cout << "data written to file " << file_basename << std::endl;
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

    std::array<string, 3> pressure_constraint_name = {"POLY_QUAD", "SET", "AVERAGE"};

    // standard options defined in definition of structs
    RBFFDOptions rbffd_opt;

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
    file_base << "Daten/" << domain_name[static_cast<int>(rbffd_opt.domain_geometry)];
    #ifdef USE_UMFPACK
    file_base << "_umfpack";
    #else
    file_base << "_eigenLU";
    #endif
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
    for (; rbffd_opt.dx_u > 1.0/120.0; rbffd_opt.dx_u/=1.2) {
        test_strategy(file_base.str(), domain_name[static_cast<int>(rbffd_opt.domain_geometry)], rbffd_opt);
    }

    uninit_h2lib();

    return 0;
}
