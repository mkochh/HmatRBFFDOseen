#ifndef OSEENCLUSTER_HEADER
#define OSEENCLUSTER_HEADER

#include  "../../H2Lib/h2lib.h"
#include <medusa/Medusa_fwd.hpp>
#include <Eigen/SparseCore>
#include <queue>
#include <vector>
#include <metis.h>
#include "../discretization/domain.hpp"
#include "../auxiliaries/aux_h2lib.h"

// useful for struct when computing velocity and pressure cluster simultaneously
struct clusterStorage {
    pcluster c_velocity;
    pcluster c_pressure;
};

enum class PartitionType {
    METIS,
    GEOM
};

enum class SeparatorType {
    SIMPLE,
    MINIMUM
};

enum class VelocityClusterType {
    STANDARD_DD,
    COUPLED_DD
};

enum class PressureClusterType {
    COUPLED_WITH_INTERFACE,
    COUPLED_NO_INTERFACE,
    UNCOUPLED_GEOM,
    COUPLED_ONE_ZERO_BLOCK,
    UNCOUPLED_METIS
};

struct ClusteringOptions {
    // Cluster options
    PartitionType partition_type = PartitionType::METIS;
    SeparatorType separator_type = SeparatorType::SIMPLE; 
    VelocityClusterType velocity_cluster_type = VelocityClusterType::STANDARD_DD; 
    PressureClusterType pressure_cluster_type = PressureClusterType::COUPLED_WITH_INTERFACE;
    int max_leaf_size_vel = 50; 
    int max_leaf_size_p = 30;
    int min_size_after_disection = 10;
    int connectivity_degree = 0; // velocity clustering based on neareast neighbor graph with given connectivity degree (if connectivity_degree > 0)
    // -1 means do not use neareast neighbor graph, if greater than 0 use nearest neighbor graph until max_depth is reached, 
    // then use velocity matrix
    int max_depth_near = -1; 
    double depth_factor = 1.0/3.0;
};

// generate a graph for partitioning from velocity discretization
Eigen::SparseMatrix<int, Eigen::RowMajor> 
createNearestNeighborMatrixVelocity(const mm::OseenDiscretizationBetter& dc, int degree = 14);

// generate a graph for partitioning from pressure discretization
Eigen::SparseMatrix<int, Eigen::RowMajor> 
createNearestNeighborMatrixPressure(const mm::OseenDiscretizationBetter& dc, int degree = 28);

// returns the block mat(idx1, idx2) of dimension size1 x size2
Eigen::SparseMatrix<int, Eigen::RowMajor> 
getSubGraph(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size1, uint* idx1, int size2, uint* idx2);

// build interface cluster using breadth first search starting from two different points, that are chosen at approximately 
// opposite ends of the graph which corresponds to the interface
pcluster 
build_bfs_interface_cluster(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size, uint* idx, 
        int clf, int dim, int level);

// build a cluster tree for the pressure using the velocity cluster tree
pcluster 
build_coupled_cluster_p(int sizep, uint* idxp, int clfp, Eigen::SparseMatrix<int, Eigen::RowMajor>& matp, 
    mm::Range<mm::Vec3d>& positions_p, pcluster c_vel);

// get clustering in a blackbox fashion only requiring the matrix, sizes of velocity and pressure unknowns and the dimension of the problem
// get domain decomposition clustering for velocity using Metis partitioning strategy and a separator strategy
// then get the pressure clustering depending only on the clustering of the velcoity
// there is no need to symmetrize mat, this will be handled inside this function
clusterStorage getBlackboxClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    int N_u, int N_p, int dim, ClusteringOptions cluster_opt);

// get domain decomposition clustering for velocity using a partitioning strategy and a separator strategy
// then get the pressure clustering depending on pressure_cluster_type
// there is no need to symmetrize mat, this will be handled inside this function
// this additionaly creates a auxiliary sparse matrix for the velocity based connectivity degree if
// cluster_opt.connectivity_degree is greater 0 (else the velocity block of mat is used)
// to get smaller separators, this is useful in combination with functions in support_by_cluster.hpp
// cg_support_size is the size of the support used when constructing the cluster geometry based on the supports
clusterStorage getClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, int cg_support_size = 0);

clusterStorage getClusteringGeom(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, int cg_support_size);

pcluster getCO_DD_VelClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_vel, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_div,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_grad, const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, pcluster rootp_given, int cg_support_size = 0);

pcluster getSubdomainClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, int n_parts, 
    const mm::OseenDiscretizationBetter& dc, const ClusteringOptions& cluster_opt);

#endif