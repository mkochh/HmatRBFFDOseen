#ifndef IO_HEADER
#define IO_HEADER

#include <medusa/Medusa_fwd.hpp>
#include <Eigen/SparseCore>
#include <string>

#include "../preconditioning/hlu.h"
#include "../../H2Lib/h2lib.h"

struct DiscretizationData {
    int N_ui = 0;
    int N_ghost = 0;
    int N_p = 0;
    int N_dofs = 0;
    double dx_u = 0.0;
    Eigen::VectorXd exact_solution;
};

struct Timings {
    double time_disc = 0.0;
    double time_con = 0.0;
    double time_supp = 0.0;
    double time_create = 0.0;
    double time_total_rbffd = 0.0;
    double time_clustering = 0.0;
    double time_prcd_setup = 0.0;
    double time_prcd = 0.0;
    double time_solve = 0.0;
    double time_prcd_solve = 0.0;
    double time_total = 0.0;
};

// useful to work with matrix in Matlab
void writeMatrix2File(Eigen::SparseMatrix<double, Eigen::RowMajor> M, std::string name = "");

// write vector and positions to a vtk file, returns 0 if succesful, 1 else
int write2vtk(Eigen::VectorXd vec, mm::Range<mm::Vec3d> positions);

// write all the data regarding discretization and solution process to a csv file
void writeData2CSV(std::string file_basename,
    const DiscretizationData& disc_data, const Eigen::VectorXd& rhs, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, 
    pavector sol_h2, pspmatrix mat_h2, Block_HLU_Prcd* P, int iter,
    const Block_HLU_Times& hlu_times, const Timings& times, const HLU_Memory& hlu_memory, const HLU_Stats hlu_stats, double tol_bicgstab);

void writeCSPdata2CSV(std::string file_basename, const DiscretizationData& disc_data, const Csp_Data& csp_data, size_t nnz_mat);

// write cluster structure (including indices) to a matlab file for visualization
void writeCluster2Matlab(pcluster root, std::string file_name);

// write cluster structure to a file, no indices
void writeClusterStructure(pcluster root, std::string file_name);

#endif