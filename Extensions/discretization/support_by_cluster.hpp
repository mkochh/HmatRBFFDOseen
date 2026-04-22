#ifndef SUPPORT_BY_CLUSTER_HEADER
#define SUPPORT_BY_CLUSTER_HEADER

#include "../../H2Lib/h2lib.h"
#include <vector>
#include "domain.hpp"
#include <Eigen/SparseCore>

// set search among and for nodes for laplacian and convection operator according to the cluster tree c
void 
setSearchAmongForNodesLapConv(pcluster c, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, int N_ub, 
        std::vector<int>& interface_idx, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

// set search among and for nodes for hyperviscosity operator according to the cluster tree c
void 
setSearchAmongForNodesHyp(pcluster c, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, int N_ub, 
        std::vector<int>& interface_idx, uint min_size);

// set search among and for nodes for gradient operator according to the cluster tree c
void 
setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, 
        int N_u, int N_ub, uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

// set search among and for nodes for divergence operator according to the cluster tree c
void 
setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, 
        int N_u, int N_ub, std::vector<int> interface_idx, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

// sets search among and for nodes in OseenDiscretizationBetter according to the cluster tree using above functions
Indices
setSearchAmongForNodesOseen(mm::OseenDiscretizationBetter& dc, pcluster rootv, pcluster rootp, 
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, int max_depth = -1);

#endif // SUPPORT_BY_CLUSTER_HEADER