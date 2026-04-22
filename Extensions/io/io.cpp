#include "io.h"
#include "../auxiliaries/aux_h2lib.h"

#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

void writeMatrix2File(Eigen::SparseMatrix<double, Eigen::RowMajor> M, std::string name)
{
    int i = 0;
    Eigen::VectorXi rows(M.nonZeros());
    Eigen::VectorXi cols(M.nonZeros());
    Eigen::VectorXd values(M.nonZeros());
    for (int k=0; k<M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M,k); it; ++it) {
            rows(i) = it.row();
            cols(i) = it.col();
            values(i) = it.value();
            i++;
        }
    }

    std::string rows_filename = "Daten/mat" + name + "_rows.txt";
    std::ofstream rows_file(rows_filename);
    for (int i = 0; i < rows.size(); i++) { rows_file << rows(i) << ";"; }
    rows_file.close();
    std::string cols_filename = "Daten/mat" + name + "_cols.txt";
    std::ofstream cols_file(cols_filename);
    for (int i = 0; i < cols.size(); i++) { cols_file << cols(i) << ";"; }
    cols_file.close();
    std::string values_filename = "Daten/mat" + name + "_values.txt";
    std::ofstream values_file(values_filename);
    values_file.precision(std::numeric_limits<double>::digits10 + 1);
    for (int i = 0; i < values.size(); i++) { values_file << values(i) << ";"; }
    values_file.close();

    std::cout << "Matrix written to files" << std::endl;
}

int write2vtk(Eigen::VectorXd vec, mm::Range<mm::Vec3d> positions) {
    // number of points
    int N = positions.size(); 

    // Create an ofstream object to manage file output
    std::ofstream outFile;

    // Open a file named "output.txt" in write mode
    outFile.open("output.vtk");

    // Check if the file was opened successfully
    if (!outFile) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // set precision: accurate to 9 decimal places
    outFile << std::setprecision(10);

    // Write formatted data to the file
    outFile << "# vtk DataFile Version 2.0\n";
    outFile << "Unstructured Grid Example\n";
    outFile << "ASCII\n\n";
    outFile << "DATASET UNSTRUCTURED_GRID\n";
    outFile << "POINTS " << N << " double\n";

    for (int i = 0; i < N; i++) {
        outFile << positions[i][0] << " " << positions[i][1] << " " << positions[i][2] << "\n";
    }

    outFile << "\nPOINT_DATA " << N << "\n";
    outFile << "VECTORS vectors double\n";
    for (int i = 0; i < N; i++) {
        outFile << vec[i] << " " << vec[i+N] << " " << vec[i+2*N] << "\n";
    }
    
    // Close the file
    outFile.close();

    // Inform the user that the data has been written
    std::cout << "Data has been written to output.vtk" << std::endl;

    return 0;
}

// print header for results csv file
static void print_header_to_file(std::string path)
{
    std::ofstream file(path);

    file << "N_ui;";
    file << "N_ghost;";
    file << "N_p;";
    file << "N_dofs;";
    file << "nnz;";
    file << "dx_u;";
    file << "iter;";
    file << "t_disc;";
    file << "t_con;";
    file << "t_supp;";
    file << "t_create;";
    file << "t_total_rbffd;";
    file << "t_prcd_setup;";
    file << "t_clustering;";
    file << "t_build_blocks;";
    file << "t_build_hmatrix;";
    file << "t_lu_vel;";
    file << "t_grad_lower;";
    file << "t_grad_upper;";
    file << "t_schur_mul;";
    file << "t_schur_computation;";
    file << "t_lu_schur;";
    file << "t_prcd;";
    file << "t_solve;";
    file << "t_prcd_solve;";
    file << "t_total;";
    file << "solution_error;";
    file << "solution_error_u;";
    file << "solution_error_p;";
    file << "solution_error_inf;";
    file << "solution_error_u_inf;";
    file << "solution_error_p_inf;";
    file << "residual_error;";
    file << "lu_vel_error;";
    file << "lu_schur_error;";
    file << "solver_tol;";
    file << "depth_vel_block;";
    file << "depth_schur_block;";
    file << "depth_grad_block;";
    file << "depth_div_block;";
    file << "C_sp_vel;";
    file << "C_sp_schur;";
    file << "C_sp_grad;";
    file << "C_sp_div;";
    file << "max_rank_vel;";
    file << "max_rank_vel_lu;";
    file << "max_ls_vel;";
    file << "max_rank_grad;";
    file << "max_rank_grad_solved;";
    file << "max_ls_grad;";
    file << "max_rank_div;";
    file << "max_rank_div_solved;";
    file << "max_ls_div;";
    file << "max_rank_schur;";
    file << "max_rank_schur_lu;";
    file << "max_ls_schur;";
    file << "h_mem_vel;";
    file << "h_mem_vel_lu;";
    file << "h_mem_grad;";
    file << "h_mem_grad_solved;";
    file << "h_mem_div;";
    file << "h_mem_div_solved;";
    file << "h_mem_schur;";
    file << "h_mem_schur_lu;";
    file << "mem_sp_mat;";

    file << std::endl;

    file.close();
}

static void print_header_to_file_csp(std::string path)
{
    std::ofstream file(path);

    file << "N_ui;";
    file << "N_ghost;";
    file << "N_p;";
    file << "N_dofs;";
    file << "nnz;";
    file << "dx_u;";
    file << "depth_vel_block;";
    file << "depth_schur_block;";
    file << "depth_grad_block;";
    file << "depth_div_block;";
    file << "C_sp_vel;";
    file << "C_sp_schur;";
    file << "C_sp_grad;";
    file << "C_sp_div;";

    file << std::endl;

    file.close();
}

void writeData2CSV(std::string file_basename,
    const DiscretizationData& disc_data, const Eigen::VectorXd& rhs, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, 
    pavector sol_h2, pspmatrix mat_h2, Block_HLU_Prcd* P, int iter,
    const Block_HLU_Times& hlu_times, const Timings& times, const HLU_Memory& hlu_memory, const HLU_Stats hlu_stats, double tol_bicgstab) {
    // Compute distance to e
    Eigen::VectorXd sol_vec = vecH2MM(sol_h2);
    Eigen::VectorXd u_exact = disc_data.exact_solution;
    int N = disc_data.N_ui + disc_data.N_ghost;
    int N_p = disc_data.N_p;
    int N_dofs = disc_data.N_dofs;
    double norm_sol = (sol_vec-u_exact).norm()/u_exact.norm();
    double norm_sol_inf = (sol_vec-u_exact).lpNorm<Eigen::Infinity>();
    double norm_sol_u = (sol_vec.head(3*N)-u_exact.head(3*N)).norm()/u_exact.head(3*N).norm();
    double norm_sol_u_inf = (sol_vec.head(3*N)-u_exact.head(3*N)).lpNorm<Eigen::Infinity>();
    double norm_sol_p = (sol_vec.segment(3*N,N_p)-u_exact.segment(3*N,N_p)).norm()/u_exact.segment(3*N,N_p).norm();
    double norm_sol_p_inf = (sol_vec.segment(3*N,N_p)-u_exact.segment(3*N,N_p)).lpNorm<Eigen::Infinity>();
    double norm_res = (rhs-mat*sol_vec).norm()/rhs.norm();
    double norm_luvel = norm2diff_lr_sparsematrix_hmatrix(mat_h2->A, P->A);
    double norm_luschur = norm2diff_lr_schurcomplement_hmatrix_rbffd(P, P->S);
    // double norm_luvel = norm2diff_id_lr_sparsematrix_hmatrix(mat_h2->A, P->A);
    // double norm_luschur = norm2diff_id_lr_schurcomplement_hmatrix_rbffd(P, P->S);

    int depth_vel_block = getdepth_block(P->block_vel);
    int depth_schur_block = getdepth_block(P->block_schur);
    int depth_grad_block = getdepth_block(P->block_grad);
    int depth_div_block = getdepth_block(P->block_div);

    int C_sp_vel = compute_csp_partition(P->block_vel->son[0]);
    int C_sp_schur = compute_csp_partition(P->block_schur->son[0]);
    int C_sp_grad = compute_csp_partition(P->block_grad->son[0]);
    int C_sp_div = compute_csp_partition(P->block_div->son[0]);

    size_t mem_sp_mat = getsize_sparsematrix(mat_h2->A) + getsize_sparsematrix(mat_h2->C);
    for (int i = 0; i < 6; i++)
        mem_sp_mat += getsize_sparsematrix(mat_h2->B[i]);

    // Print data to file
    std::stringstream outfile_name;
    outfile_name << file_basename << ".csv";
    // if file doesn't exist yet write header
    if (!std::ifstream(outfile_name.str()))
        print_header_to_file(outfile_name.str());

    std::ofstream outfile(outfile_name.str(), std::ofstream::app);
    auto default_precision = outfile.precision();

    outfile << N << ";" << disc_data.N_ghost << ";" << N_p << ";" << N_dofs << ";" << mat.nonZeros() << ";" << disc_data.dx_u << ";" << iter << ";";

    outfile.setf(std::ios::fixed);
    outfile.precision(2);
    outfile << times.time_disc << ";" << times.time_con << ";" << times.time_supp << ";" << times.time_create << ";" << times.time_total_rbffd << ";";
    outfile << times.time_prcd_setup << ";" << times.time_clustering << ";" << hlu_times.build_blocks << ";" << hlu_times.build_hmatrix << ";" << hlu_times.velocity_lu << ";" << hlu_times.grad_lower_solve << ";";
    outfile << hlu_times.grad_upper_solve << ";" << hlu_times.grad_schur_multiplication << ";" << hlu_times.schur_computation << ";" << hlu_times.schur_lu << ";";
    outfile << times.time_prcd << ";" << times.time_solve << ";" << times.time_prcd_solve << ";" << times.time_total << ";";
    outfile.unsetf(std::ios::fixed);
    outfile.precision(default_precision);

    outfile.setf(std::ios::scientific);
    outfile << norm_sol << ";" << norm_sol_u << ";" << norm_sol_p << ";" 
    << norm_sol_inf << ";" << norm_sol_u_inf << ";" << norm_sol_p_inf << ";"
    << norm_res << ";" << norm_luvel << ";" << norm_luschur << ";" << tol_bicgstab << ";";
    outfile.unsetf(std::ios::scientific);

    outfile << depth_vel_block << ";" << depth_schur_block << ";" << depth_grad_block << ";" << depth_div_block << ";";
    outfile << C_sp_vel << ";" << C_sp_schur << ";" << C_sp_grad << ";" << C_sp_div << ";";
    outfile << hlu_stats.max_rank_vel << ";" << hlu_stats.max_rank_vel_lu << ";" << hlu_stats.max_leaf_size_vel << ";";
    outfile << hlu_stats.max_rank_grad << ";" << hlu_stats.max_rank_grad_solved << ";" << hlu_stats.max_leaf_size_grad << ";";
    outfile << hlu_stats.max_rank_div << ";" << hlu_stats.max_rank_div_solved << ";" << hlu_stats.max_leaf_size_div << ";";
    outfile << hlu_stats.max_rank_schur << ";" << hlu_stats.max_rank_schur_lu << ";" << hlu_stats.max_leaf_size_schur << ";";
    outfile << hlu_memory.vel << ";" << hlu_memory.vel_lu << ";" << hlu_memory.grad << ";" << hlu_memory.grad_solved << ";";
    outfile << hlu_memory.div << ";" << hlu_memory.div_solved << ";" << hlu_memory.schur << ";" << hlu_memory.schur_lu << ";";
    outfile << mem_sp_mat << ";";
    outfile << std::endl;

    outfile.close();

    std::cout << "data written to file " << file_basename << "\n" << std::endl;
}

void writeCSPdata2CSV(std::string file_basename,
    const DiscretizationData& disc_data, const Csp_Data& csp_data, size_t nnz_mat) {

    int N = disc_data.N_ui + disc_data.N_ghost;
    int N_p = disc_data.N_p;
    int N_dofs = disc_data.N_dofs;

    // Print data to file
    std::stringstream outfile_name;
    outfile_name << file_basename << ".csv";
    // if file doesn't exist yet write header
    if (!std::ifstream(outfile_name.str()))
        print_header_to_file_csp(outfile_name.str());

    std::ofstream outfile(outfile_name.str(), std::ofstream::app);

    outfile << N << ";" << disc_data.N_ghost << ";" << N_p << ";" << N_dofs << ";" << nnz_mat<< ";" << disc_data.dx_u << ";";

    outfile << csp_data.depth_vel << ";" << csp_data.depth_schur << ";" << csp_data.depth_grad << ";" << csp_data.depth_div << ";";
    outfile << csp_data.csp_vel << ";" << csp_data.csp_schur << ";" << csp_data.csp_grad << ";" << csp_data.csp_div << ";";
    outfile << std::endl;

    outfile.close();

    std::cout << "data written to file " << file_basename << "\n" << std::endl;
}

static void writeCluster2Matlab_recursion(pcluster root, std::ofstream& file, std::string prefix = "idx")
{
    if (root->sons > 0) {
        for (uint i = 0; i < root->sons; i++) {
            file << prefix << i <<" = [";
            for (uint j = 0; j < root->son[i]->size; j++) {
                file << root->son[i]->idx[j] << ", ";
            }
            file << "];\n";
            writeCluster2Matlab_recursion(root->son[i], file, prefix + std::to_string(i));
        }
    }
}

void writeCluster2Matlab(pcluster root, std::string file_name)
{
    std::ofstream file(file_name);
    writeCluster2Matlab_recursion(root, file);
    file.close();

    std::cout << "Cluster written to " << file_name << std::endl;
}

static void writeClusterStructure_recursion(pcluster root, std::ofstream& file, std::string prefix = "")
{
    if (root->sons > 0) {
        for (uint i = 0; i < root->sons; i++) {
            file << prefix << i <<", ";
            writeClusterStructure_recursion(root->son[i], file, prefix + std::to_string(i));
        }
    }
}

void writeClusterStructure(pcluster root, std::string file_name)
{
    std::ofstream file(file_name);
    writeClusterStructure_recursion(root, file);
    file.close();

    std::cout << "Cluster written to " << file_name << std::endl;
}
