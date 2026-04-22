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

    file << "nu;";
    file << "N_ui;";
    file << "N_p;";
    file << "N_total;";
    file << "t_solve;";
    file << "norm_l_inf;";
    file << "inv_norm_l_inf;";
    file << "cond;";

    file << std::endl;

    file.close();
}

void writeData2CSV(string file_basename,
    const OseenDiscretizationBetter& dc, const Eigen::VectorXd& rhs, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, 
    const double time_solve, double norm_l_inf, double inv_norm_l_inf, double cond) {
    // Compute distance to e
    int N = dc.N_ui + dc.N_uneu;
    int N_p = dc.N_p;

    // Print data to file
    std::stringstream outfile_name;
    outfile_name << file_basename << ".csv";
    // if file doesn't exist yet write header
    if (!std::ifstream(outfile_name.str()))
        print_header_to_file(outfile_name.str());

    std::ofstream outfile(outfile_name.str(), std::ofstream::app);
    auto default_precision = outfile.precision();

    outfile << dc.nu << ";" << N << ";" << N_p << ";" << dc.N_dofs << ";";

    outfile.setf(std::ios::fixed);
    outfile.precision(2);
    outfile << time_solve << ";";
    outfile.unsetf(std::ios::fixed);
    outfile.precision(default_precision);

    outfile.setf(std::ios::scientific);
    outfile << norm_l_inf << ";" << inv_norm_l_inf << ";" << cond << ";" << std::endl;
    outfile.unsetf(std::ios::scientific);

    outfile.close();
}

// solve oseen directly
void test_strategy(string file_basename, string domain_name, RBFFDOptions rbffd_opt)
{
    // set input file for polyhedron
    std::stringstream OFF_File_base;
    OFF_File_base << "OFF_Files/" << domain_name << ".off";
    string OFF_file = OFF_File_base.str();
    PolyhedronShape<Vec3d> shape = PolyhedronShape<Vec3d>::fromOFF(OFF_file);

    // construct domains needed for discretization of Oseen equations and initialize parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    OseenDiscretizationBetter dc(shape, rbffd_opt); 
    
    // determine pressure constraint %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (rbffd_opt.pressure_constraint == PressureConstraint::POLY_QUAD && dc.idxs_neu.size() == 0)
        dc.setConstraint(OFF_file); // set pressure constraint, needed to make compute pressure at high accuracy, if there are no neumann boundary nodes

    // determine supports (stencils) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dc.determineSupports();
    if (dc.use_hyperviscosity)
        dc.determineSupportsHyperViscosity();

    std::cout << "N_ui = " << dc.N_ui << " and N_p = " << dc.N_p << std::endl;
    std::cout << "N_dir = " << dc.N_ub << " and N_neu = " << dc.N_uneu << std::endl;

    // compute weights and remove dirichlet boundary conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat(dc.N_dofs,dc.N_dofs);
    Eigen::VectorXd rhs(dc.N_dofs);
    createMatrixAndRHS(mat, rhs, dc, rbffd_opt.pressure_constraint);

    Eigen::SparseMatrix<double, Eigen::ColMajor> mat_col = mat;
    
    // Estimate Condition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    auto t6 = high_resolution_clock::now();
    
    double norm_l_inf = (mat.cwiseAbs() * Eigen::VectorXd::Ones(mat.cols())).maxCoeff(); // ||matrix||_1 = max_j sum_i |a_ij|
    prn(norm_l_inf);
    Eigen::MatrixXd temp = mat.transpose();
    Eigen::PartialPivLU<Eigen::MatrixXd> solver;
    solver.compute(temp);
    double inv_norm_l_inf = Eigen::internal::rcond_invmatrix_L1_norm_estimate(solver);
    prn(inv_norm_l_inf);
    
    double cond = norm_l_inf*inv_norm_l_inf;

    auto t7 = high_resolution_clock::now();

    // post-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    duration<double> duration_solve = t7 - t6;
    double time_solve = duration_solve.count();

    writeData2CSV(file_basename, dc, rhs, mat, time_solve, norm_l_inf, inv_norm_l_inf, cond);
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
