#include "oseencluster.h"
#include <array>

Eigen::SparseMatrix<int, Eigen::RowMajor> createNearestNeighborMatrixVelocity(const mm::OseenDiscretizationBetter& dc, int degree)
{
    mm::DomainDiscretization<mm::Vec3d> d_u = dc.d_u_int;
    int N = d_u.size();
    std::vector<int> forNodes(dc.idxs_ui.size());
    for (int i = 0; i < dc.idxs_ui.size(); i++) {
        forNodes[i] = dc.idxs_ui[i]-dc.N_ub;
    }
    mm::FindClosest supp(degree);
    supp.forNodes(forNodes);
    d_u.findSupport(supp);
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat(N, N);

    std::vector<Eigen::Triplet<int>> tripletList;

    if (dc.idxs_neu.size() == 0) {
        // only dirichlet boundary nodes 
        tripletList.reserve(2*degree*N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d_u.supports()[i].size(); ++j) {
                tripletList.push_back(Eigen::Triplet<int>(i, d_u.support(i)[j], 1));
                tripletList.push_back(Eigen::Triplet<int>(d_u.support(i)[j], i, 1));
            }
        }

        mat.setFromTriplets(tripletList.begin(), tripletList.end());
    } else {
        // dirichlet and neumann boundary nodes
        tripletList.reserve(degree*N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d_u.supports()[i].size(); ++j) {
                tripletList.push_back(Eigen::Triplet<int>(i, d_u.support(i)[j], 1));
            }
        }

        Eigen::SparseMatrix<int, Eigen::RowMajor> temp(N, N);
        temp.setFromTriplets(tripletList.begin(), tripletList.end());

        // make sure rows for neumann nodes and ghost nodes have same structure and resulting matrix is symmetric
        tripletList.clear();
        tripletList.reserve(2*degree*N);
        for (int i = 0; i < temp.outerSize(); ++i) {
            for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(temp,i); it; ++it) {
                if (it.row() < dc.N_uneu && it.col() < dc.N_uneu) {
                    assert(it.row() + dc.N_ui < N && it.col() + dc.N_ui < N && "error in upper left block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()+dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()+dc.N_ui, it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()+dc.N_ui, it.col()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()+dc.N_ui, it.row()+dc.N_ui, 1));
                } else if (it.row() < dc.N_uneu && it.col() >= dc.N_ui) {
                    assert(it.row() + dc.N_ui < N && it.col() - dc.N_ui >= 0 && "error in upper right block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()+dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()+dc.N_ui, it.col()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row()+dc.N_ui, 1));
                } else if (it.row() >= dc.N_ui && it.col() < dc.N_uneu) {
                    assert(it.row() - dc.N_ui >= 0 && it.col() + dc.N_ui < N && "error in lower left block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()-dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()-dc.N_ui, it.col()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()+dc.N_ui, it.row()-dc.N_ui, 1));
                } else if (it.row() >= dc.N_ui && it.col() >= dc.N_ui) {
                    assert(it.row() - dc.N_ui >= 0 && it.col() - dc.N_ui >= 0 && "error in lower right block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()-dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()-dc.N_ui, it.col()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row()-dc.N_ui, 1));
                } else if (it.row() < dc.N_uneu && it.col() >= dc.N_uneu && it.col() < dc.N_ui) {
                    assert(it.row() + dc.N_ui < N && "error in upper block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()+dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()+dc.N_ui, 1));
                } else if (it.row() >= dc.N_uneu && it.row() < dc.N_ui && it.col() < dc.N_uneu) {
                    assert(it.col() + dc.N_ui < N && "error in left block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()+dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()+dc.N_ui, it.row(), 1));
                } else if (it.row() >= dc.N_uneu && it.row() < dc.N_ui && it.col() >= dc.N_ui) {
                    assert(it.col() - dc.N_ui >= 0 && "error in right block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col()-dc.N_ui, 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col()-dc.N_ui, it.row(), 1));
                } else if (it.row() >= dc.N_ui && it.col() >= dc.N_uneu && it.col() < dc.N_ui) {
                    assert(it.row() - dc.N_ui >= 0 && "error in lower block");
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.row()-dc.N_ui, it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row()-dc.N_ui, 1));
                } else {
                    tripletList.push_back(Eigen::Triplet<int>(it.row(), it.col(), 1));
                    tripletList.push_back(Eigen::Triplet<int>(it.col(), it.row(), 1));
                }
            }
        }

        mat.setFromTriplets(tripletList.begin(), tripletList.end());   
    }

    mat.makeCompressed();

    return mat;
}

Eigen::SparseMatrix<int, Eigen::RowMajor> createNearestNeighborMatrixPressure(const mm::OseenDiscretizationBetter& dc, int degree)
{
    mm::DomainDiscretization<mm::Vec3d> d = dc.d_u_int;
    mm::Range<int> idxp = d.addNodes(dc.d_p);
    mm::FindClosest supp(degree);
    supp.forNodes(idxp).searchAmong(dc.d_u_int.all()).forceSelf(false);
    d.findSupport(supp);
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat(dc.N_p, dc.d_u_int.size());

    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(degree*dc.N_p);
    for (int i = 0; i < dc.N_p; ++i) {
        for (int j = 0; j < d.supports()[i+dc.d_u_int.size()].size(); ++j) {
            tripletList.push_back(Eigen::Triplet<int>(i, d.support(i+dc.d_u_int.size())[j], 1));
        }
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

static Eigen::SparseMatrix<int, Eigen::RowMajor> createNearestNeighborMatrixPressureGrad(const mm::OseenDiscretizationBetter& dc, int degree)
{
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(degree*dc.d_u_int.size());
    for (int i = dc.N_ub; i < dc.d_u.size(); ++i) {
        for (int j = 0; j < dc.d_grad.supports()[i].size(); ++j) {
            tripletList.push_back(Eigen::Triplet<int>(dc.d_grad.support(i)[j]-dc.d_u.size(), i-dc.N_ub, 1));
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> mat(dc.d_p.size(), dc.d_u_int.size());
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<int, Eigen::RowMajor> 
getSubGraph(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
			int size1, uint* idx1, 
			int size2, uint* idx2)
{
    // Eigen::SparseMatrix<int, Eigen::ColMajor> temp(size1, mat.cols());
    // temp.reserve(Eigen::VectorXi::Constant(mat.cols(), std::min(static_cast<int>(2*mat.nonZeros()/mat.cols()), size1)));
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(size1*(mat.nonZeros()/mat.rows()));
    std::vector<int> flag(mat.cols(), 0);
    for (int i = 0; i < size2; i++)
        flag[idx2[i]] = 1;

    for (int i = 0; i < size1; i++) {
        for (int j = mat.outerIndexPtr()[idx1[i]]; j < mat.outerIndexPtr()[idx1[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (flag[neighbor] == 1) {
                tripletList.emplace_back(i, neighbor, 1);
                // temp.insert(i, neighbor) = 1;
            }
        }
    }

    Eigen::SparseMatrix<int, Eigen::ColMajor> temp(size1, mat.cols());
    temp.setFromTriplets(tripletList.begin(), tripletList.end());
    temp.makeCompressed();
    tripletList.clear();
    tripletList.reserve(size1*(temp.nonZeros()/temp.rows()));
    // Eigen::SparseMatrix<int, Eigen::RowMajor> sub_mat(size1, size2);
    // sub_mat.reserve(Eigen::VectorXi::Constant(size1, std::min(static_cast<int>(2*temp.nonZeros()/temp.rows()), size2)));
    for (int i = 0; i < size2; i++) {
        for (int j = temp.outerIndexPtr()[idx2[i]]; j < temp.outerIndexPtr()[idx2[i] + 1]; j++) {
            int neighbor = temp.innerIndexPtr()[j];
            tripletList.emplace_back(neighbor, i, 1);
            // sub_mat.insert(neighbor, i) = 1;
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> sub_mat(size1, size2);
    sub_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    sub_mat.makeCompressed();
    return sub_mat;
}

struct Graph {
    std::vector<int> outerStarts;
    std::vector<int> innerIndices;
    int rows;
    int cols;
};

static Graph 
getSubGraph_fast(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
			int size_row, uint* idx_row, 
			int size_col, uint* idx_col)
{
    Graph graph;
    graph.outerStarts.resize(size_row + 1, 0);
    graph.innerIndices.reserve(size_row*(mat.nonZeros()/mat.rows()));
    graph.rows = size_row;
    graph.cols = size_col;
    std::vector<int> flag(mat.cols(), 0);
    std::vector<int> inv_idx(mat.cols(), -1);
    for (int i = 0; i < size_col; i++)
        inv_idx[idx_col[i]] = i;
    for (int i = 0; i < size_col; i++)
        flag[idx_col[i]] = 1;

    int counter = 0;
    for (int i = 0; i < size_row; i++) {
        for (int j = mat.outerIndexPtr()[idx_row[i]]; j < mat.outerIndexPtr()[idx_row[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (flag[neighbor] == 1) {
                graph.innerIndices.push_back(inv_idx[neighbor]);
                counter++;
            }
        }
        graph.outerStarts[i+1] = counter;
    }

    return graph;
}

// same as getSubGraph but assumes idx1 = idx2 and mat is symmetric
static Graph 
getSubGraphSym(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
			int size, uint* idx)
{
    Graph graph;
    graph.outerStarts.resize(size + 1, 0);
    graph.innerIndices.reserve(size*(mat.nonZeros()/mat.rows()));
    graph.rows = size;
    graph.cols = mat.cols();
    std::vector<int> flag(mat.cols(), 0);
    std::vector<int> inv_idx(mat.rows(), -1);
    for (int i = 0; i < size; i++)
        inv_idx[idx[i]] = i;
    for (int i = 0; i < size; i++)
        flag[idx[i]] = 1;

    int counter = 0;
    for (int i = 0; i < size; i++) {
        for (int j = mat.outerIndexPtr()[idx[i]]; j < mat.outerIndexPtr()[idx[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (flag[neighbor] == 1) {
                graph.innerIndices.push_back(inv_idx[neighbor]);
                counter++;
            }
        }
        graph.outerStarts[i+1] = counter;
    }

    return graph;
}

/***
 * Functions to partition the interface that arises domain decomposition clustering
 */

namespace {

// depth first search to check if the matrix is connected
bool checkConnectedness(Graph& graph)
{
    // Check if the matrix is connected
    std::vector<int> visited(graph.rows, 0);
    std::vector<int> stack;
    stack.reserve(graph.rows);
    stack.push_back(0);
    visited[0] = 1;
    while (!stack.empty()) {
        int node = stack.back();
        stack.pop_back();
        for (int j = graph.outerStarts[node]; j < graph.outerStarts[node + 1]; j++) {
            int neighbor = graph.innerIndices[j];
            if (visited[neighbor] == 0) {
                visited[neighbor] = 1;
                stack.push_back(neighbor);
            }
        }
    }
    for (int i = 0; i < graph.rows; i++) {
        if (visited[i] == 0) {
            return false;
        }
    }
    return true;
}

// find neighbors of the idxs neighbors in the graph mat and add them to neighbors
void findNeighbors(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, std::vector<uint>& neighbors)
{
    std::vector<int> visited(mat.rows(), 0);
    size_t size = neighbors.size();
    for (size_t i = 0; i < size; i++)
        visited[neighbors[i]] = 1;
    for (size_t i = 0; i < size; i++) {
        for (int j = mat.outerIndexPtr()[neighbors[i]]; j < mat.outerIndexPtr()[neighbors[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (visited[neighbor] == 0) {
                visited[neighbor] = 1;
                neighbors.push_back(neighbor);
            }
        }
    }
}

// find a graph that surrounds the initial graph and is connected
Graph findSurroundingGraph(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, std::vector<uint>& neighbors)
{
    Graph surroundingGraph = getSubGraph_fast(mat, neighbors.size(), neighbors.data(), neighbors.size(), neighbors.data());
    // will be true eventually because mat is fully connected
    while(!checkConnectedness(surroundingGraph)) {
        findNeighbors(mat, neighbors);
        surroundingGraph = getSubGraph_fast(mat, neighbors.size(), neighbors.data(), neighbors.size(), neighbors.data());
    }

    return surroundingGraph;
}

// find two start nodes to perform bfs patitioning afterwards
// the first size rows of the matrix are expected to correspond to the seperator
std::vector<int> bfsFindStart(Graph& graph, int size)
{
    int N = graph.cols;
    int N_bfs = 2;
    int start = 0; // start node
    std::vector<int> res(2,0); // vector storing the two start nodes for bfs partitioning
    for (int i = 0; i < N_bfs; i++)
    {
        std::vector<int> visited(N, 0);
        std::queue<int> q;
        visited[start] = 1;
        q.push(start);
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            for (int j = graph.outerStarts[node]; j < graph.outerStarts[node + 1]; j++) {
                int neighbor = graph.innerIndices[j];
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    q.push(neighbor);
                    if (neighbor < size) {
                        res[(i+1)%2] = neighbor;
                    }
                }
            }
        }

        start = res[(i+1)%2];
    }

    return res;
}

// partition the interface using breadth first search
// the first size rows of the matrix are expected to correspond to the seperator
std::vector<int> bfsPartInterface(Graph& graph, int size)
{
    std::vector<int> start = bfsFindStart(graph, size);
    std::vector<int> remaining_sep(size);
    for (int i = 0; i < size; i++) {
        remaining_sep[i] = i;
    }
    std::vector<int> partition(size);
    std::vector<int> visited(graph.rows, 0);
    std::queue<int> q1;
    visited[start[0]] = 1;
    q1.push(start[0]);
    partition[start[0]] = 0;
    remaining_sep.erase(std::remove(remaining_sep.begin(), remaining_sep.end(), start[0]), remaining_sep.end());
    std::queue<int> q2;
    visited[start[1]] = 1;
    q2.push(start[1]);
    partition[start[1]] = 1;
    remaining_sep.erase(std::remove(remaining_sep.begin(), remaining_sep.end(), start[1]), remaining_sep.end());
    while ((!q1.empty() && !remaining_sep.empty()) || (!q2.empty() && !remaining_sep.empty())) {
        if (!q1.empty()) {
            int node = q1.front();
            q1.pop();
            for (int j = graph.outerStarts[node]; j < graph.outerStarts[node + 1]; j++) {
                int neighbor = graph.innerIndices[j];
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    q1.push(neighbor);
                    if (neighbor < size) {
                        partition[neighbor] = 0;
                        remaining_sep.erase(std::remove(remaining_sep.begin(), remaining_sep.end(), neighbor), remaining_sep.end());
                    }
                }
            }
        }
        if (!q2.empty() && !remaining_sep.empty()) {
            int node = q2.front();
            q2.pop();
            for (int j = graph.outerStarts[node]; j < graph.outerStarts[node + 1]; j++) {
                int neighbor = graph.innerIndices[j];
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    q2.push(neighbor);
                    if (neighbor < size) {
                        partition[neighbor] = 1;
                        remaining_sep.erase(std::remove(remaining_sep.begin(), remaining_sep.end(), neighbor), remaining_sep.end());
                    }
                }
            }
        }
    }

    return partition;
}

}

pcluster build_bfs_interface_cluster(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size, uint* idx, 
        int clf, int dim, int level)
{
    pcluster c;

    if (size > clf) {
        if (level % dim) {
            // graph for the seperator is likely not connected, therefore find surrounding graph that is connected
            std::vector<uint> idx_vec(size);
            for (int i = 0; i < size; i++)
                idx_vec[i] = idx[i];
            // first size nodes in surrounding graph correspond to idxs of the seperator
            Graph surroundingGraph = findSurroundingGraph(mat, idx_vec);
            std::vector<int> partition = bfsPartInterface(surroundingGraph, size);

            int size0 = 0, size1 = 0;
            level++;
            
            for (int i = 0; i < size; i++) {
                if (partition[i] == 0) {
                    int tmp = idx[i];
                    idx[i] = idx[size0];
                    idx[size0] = tmp;
                    size0++;
                } else {
                    size1++;
                }
            }

            if (size0 > 0) {
                if (size1 > 0) {
                    c = new_cluster(size, idx, 2, dim);
                    c->son[0] = build_bfs_interface_cluster(mat, size0, idx, clf, dim, level);
                    c->son[1] = build_bfs_interface_cluster(mat, size1, idx + size0, clf, dim, level);
                } else {
                    // did not find partition into two sets
                    assert(size0 > 0);
                    assert(size1 == 0);
                    c = new_cluster(size, idx, 0, dim);
                }
            } else {
                // did not find partition into two sets
                assert(size0 == 0);
                assert(size1 > 0);
                c = new_cluster(size, idx, 0, dim);
            }
        } else {
            // level%dim == 0
            level++;
            c = new_cluster(size, idx, 1, dim);
            c->son[0] = build_bfs_interface_cluster(mat, size, idx, clf, dim, level);
        }
    } else {
        // size <= clf
        c = new_cluster(size, idx, 0, dim);
    }

    c->type = 2; // interface cluster
    update_cluster(c);

    return c;
}

/***
 * Functions to partition and sort the velocity and pressure indices
 */

namespace {

// sort idx of size according to partition into two sets, store respective sizes in size0 and size1
void sortByPartition(std::vector<int>& partition, uint* idx, int size, int& size0, int& size1)
{
    // sort to have partition 0 first then partition 1
    for (int i = 0; i < size; i++) {
        if (partition[i] == 0) {
            int tmp = idx[i];
            idx[i] = idx[size0];
            idx[size0] = tmp;
            size0++;
        } else {
            size1++;
        }
    }
}

// forces idx that are marked in flag_neu to be in the same partition
// size is the size of idx
// e.g. flag_neu = [0, 0, 1, 1, 0, 0, 0, 2, 2]
// then N_offset = 6, partition is the cuurent partition before forcing flaged indexes to be in the same partition
// 2 and 7 as well as 3 and 8 will be forced into the same partition
// it is assumed that if i is in idx then so is i+N_offset
void keepInSamePartition(uint* idx, std::vector<int> flag_neu, std::vector<int>& part, int size, int N_offset)
{
    std::vector<int> temp(flag_neu.size(), 0);
    for (int i = 0; i < size; i++) {
        // neumann node at idx[i]
        if (flag_neu[idx[i]] == 1) {
            flag_neu[idx[i]+N_offset] = 3;
            temp[idx[i]+N_offset] = part[i];
        }
        // ghost node at idx[i]
        if (flag_neu[idx[i]] == 2) {
            flag_neu[idx[i]-N_offset] = 3;
            temp[idx[i]-N_offset] = part[i];
        }
        // ghost or neumann node whose partner has been visited before
        if (flag_neu[idx[i]] == 3) {
            part[i] = temp[idx[i]];
        }
    }
}

// partition a graph into nParts using Metis
std::vector<int> getPartitionKwayMetis(Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int nParts, int* options = nullptr) {
    int nVertices = mat.rows(); // number of vertices
    int nCon = 1; // number of constraints
    int objval;
    std::vector<int> partition(nVertices, 0); // partition vector
    
    // Call METIS to partition the matrix
    int status = METIS_PartGraphKway(&nVertices, &nCon, mat.outerIndexPtr(), mat.innerIndexPtr(), nullptr, nullptr, nullptr,
                                       &nParts, nullptr, nullptr, options, &objval, partition.data());

    assert(status == METIS_OK && "METIS partitioning failed");
    
    return partition;
}

// partition a graph into nParts using Metis
static std::vector<int> getPartitionKwayMetis(Graph& graph, int nParts, int* options = nullptr) {
    int nVertices = graph.rows; // number of vertices
    int nCon = 1; // number of constraints
    int objval;
    std::vector<int> partition(nVertices, 0); // partition vector
    
    // Call METIS to partition the matrix
    int status = METIS_PartGraphKway(&nVertices, &nCon, graph.outerStarts.data(), graph.innerIndices.data(), nullptr, nullptr, nullptr,
                                       &nParts, nullptr, nullptr, options, &objval, partition.data());

    assert(status == METIS_OK && "METIS partitioning failed");
    
    return partition;
}

// partition velocity using metis, then sort idx according to partition, size0 and size1 are updated
std::vector<int> partitionVelocityMetis(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size, uint* idx, int nParts = 2)
{
    if (size < mat.rows()) {
        Graph graph = getSubGraphSym(mat, size, idx);
        return getPartitionKwayMetis(graph, nParts);
    } else {
        assert(size == mat.rows());
        Eigen::SparseMatrix<int, Eigen::RowMajor> mat_copy = mat;
        return getPartitionKwayMetis(mat_copy, nParts);
    }
}

std::vector<int> partitionVelocityGeom(pclustergeometry cg, int size, uint* idx)
{
    update_point_bbox_clustergeometry(cg, static_cast<uint>(size), idx);
    
    /* compute the direction of partition , choose direction with largest extend*/
    uint direction = 0;
    double a = cg->hmax[0] - cg->hmin[0]; // extend of cluster bounding box in direction 0

    for (uint j = 1; j < cg->dim; j++) {
        double m = cg->hmax[j] - cg->hmin[j]; // extend of cluster bounding box in direction j
        if (a < m) {
            a = m;
            direction = j;
        }
    }

    double middle = (cg->hmax[direction] + cg->hmin[direction]) / 2.0;

    std::vector<int> partition(size, 0);
    for (int i = 0; i < size; i++) {
        if (cg->x[idx[i]][direction] > middle) {
            partition[i] = 1;
		}
    }

    return partition;
}

void sortVelocity(std::vector<int>& partition, const std::vector<int> flag_neu, uint* idx, int size, int& size0, int& size1, int N_neu)
{
    // keep neumann and its corresponding ghost node in the same partition, if there are any neumann nodes
    // assumes that neumann node and corresponding ghost node are offset by N_neu
    if (N_neu)
        keepInSamePartition(idx, flag_neu, partition, size, N_neu);
    // sort to have partition 0 first then partition 1
    sortByPartition(partition, idx, size, size0, size1);
}

// partition rows (corresponds to pressure nodes) according to connections to columns (which are already partitioned velocity nodes)
std::vector<int> getPartitionPressure(Graph& graph_p_part, int size0, int size1, int size_p)
{
    std::vector<int> partition(size_p, 0);
    // find partitioning into two sets
    for (int i = 0; i < size_p; i++) {
        int d0 = 0; int d1 = 0;
        for (int j = graph_p_part.outerStarts[i]; j < graph_p_part.outerStarts[i + 1]; j++) {
            int neighbor = graph_p_part.innerIndices[j];
            if (neighbor < size0) {
                d0++;
            } else if (neighbor < size0 + size1) {
                d1++;
            } else {
                break;
            }
        }
        if (d0 > d1) {
            partition[i] = 0;
        } else if (d1 > d0) {
            partition[i] = 1;
        } else {
            partition[i] = i%2; // if d0 = d1 then assign partition randomly
        }
    }

    return partition;
}

// partition pressure according to connections to velocity nodes, then sort idxp according to partition, size0_p and size1_p are updated
void partitionAndSortPressure(Eigen::SparseMatrix<int, Eigen::RowMajor>& mat_p, int size0, int size1, uint* idx, 
        int& size0_p, int& size1_p, int size_p, uint* idxp)
{   
    Graph graph_p_part = getSubGraph_fast(mat_p, size_p, idxp, size0+size1, idx);
    // partition pressure according to whether they have more connections to velocity partition 0 or 1
    std::vector<int> partition_p = getPartitionPressure(graph_p_part, size0, size1, size_p);
    sortByPartition(partition_p, idxp, size_p, size0_p, size1_p);
}

// determine bounding box of cluster c from positions
// only uses information on positions of points not the support of the cluster
// needed to define weak or strong admissibility condition for schur complement
void determineBoundingBoxFromPoints(pcluster c, mm::Range<mm::Vec3d>& positions)
{
    for (uint j = 0; j < c->dim; j++) {
        c->bmin[j] = positions[c->idx[0]][j];
        c->bmax[j] = positions[c->idx[0]][j];
    }

    for (uint i = 1; i < c->size; i++) {
        for (uint j = 0; j < c->dim; j++) {
            if (positions[c->idx[i]][j] < c->bmin[j]) {
                c->bmin[j] = positions[c->idx[i]][j];
            }
            if (positions[c->idx[i]][j] > c->bmax[j]) {
                c->bmax[j] = positions[c->idx[i]][j];
            }
        }
    }
}

}

/* 
 * Functions to compute separators for the domain decomposition
 */

namespace {

void MinCover_ColDFS(int* xadj, int* adjncy, int root, std::vector<int>& mate, std::vector<int>& where, int flag)
{
    if (flag == 10) {
        if (where[root] == 3)
            return;
        where[root] = 3;
        for (int i = xadj[root]; i < xadj[root + 1]; i++)
            MinCover_ColDFS(xadj, adjncy, adjncy[i], mate, where, 20);
    } else {
        if (where[root] == 6)
            return;
        where[root] = 6;
        if (mate[root] != -1)
            MinCover_ColDFS(xadj, adjncy, mate[root], mate, where, 10);
    }
}

static void MinCover_RowDFS(int* xadj, int* adjncy, int root, std::vector<int>& mate, std::vector<int>& where, int flag)
{
    if (flag == 20) {
        if (where[root] == 4)
            return;
        where[root] = 4;
        for (int i = xadj[root]; i < xadj[root + 1]; i++)
            MinCover_RowDFS(xadj, adjncy, adjncy[i], mate, where, 10);
    } else {
        if (where[root] == 1)
            return;
        where[root] = 1;
        if (mate[root] != -1)
            MinCover_RowDFS(xadj, adjncy, mate[root], mate, where, 20);
    }
}

// returns the mininmal vertex cover of a bipartite graph for a given maximum matching, mostly used in MinCover
// see: Computing the Block Triangular Form of a Sparse Matrix ALEX POTHEN and CHIN-JU FAN
std::vector<int> MinCover_Decompose(int* xadj, int* adjncy, int asize, int bsize, std::vector<int>& mate)
{
    std::vector<int> where(bsize);
    std::vector<int> cover; cover.reserve(bsize);
    std::array<int, 10> card;

    
    for (int i = 0; i < 10; i++)
        card[i] = 0;
    for (int i = 0; i < asize; i++)
        where[i] = 2;
    for (int i = asize; i < bsize; i++)
        where[i] = 5;
    
    for (int i = 0; i < asize; i++) {
        if (mate[i] == -1)
            MinCover_ColDFS(xadj, adjncy, i, mate, where, 10);
    }
    for (int i = asize; i < bsize; i++) {
        if (mate[i] == -1)
            MinCover_RowDFS(xadj, adjncy, i, mate, where, 20);
    }

    for (int i = 0; i < bsize; i++)
        card[where[i]]++;

    if (std::abs(card[1] + card[2] - card[6]) < std::abs(card[1] - card[5] - card[6])) {
        for (int i = 0; i < bsize; i ++) {
            if (where[i] == 1 || where[i] == 2 || where[i] == 6)
                cover.push_back(i);
        }
    } else {
        for (int i = 0; i < bsize; i ++) {
            if (where[i] == 1 || where[i] == 5 || where[i] == 6)
                cover.push_back(i);
        }
    }

    return cover;
}

// augment a existing matching using alternating paths
int MinCover_Augment(int* xadj, int* adjncy, int col, std::vector<int>& mate, std::vector<int>& flag, std::vector<int>& level, int maxlevel)
{
    int status; 
    flag[col] = 2;
    for (int i=xadj[col]; i<xadj[col+1]; i++) {
        int row = adjncy[i];

        if (flag[row] == 1) { /* First time through this row node */
            if (level[row] == maxlevel) {  /* (col, row) is an edge of the G^T */
                flag[row] = 2;  /* Mark this node as being visited */
                if (maxlevel != 0)
                    status = MinCover_Augment(xadj, adjncy, mate[row], mate, flag, level, maxlevel-1);
                else
                    status = 1;

                if (status) {
                    mate[col] = row;
                    mate[row] = col;
                    return 1;
                }
            }
        }
    }

    return 0;
}

/*************************************************************************
* This function returns the min-cover of a bipartite graph.
* The algorithm used is due to Hopcroft and Karp as modified by Duff etal
* following the implementation in metis mincover.c
* adj: the adjacency list of the bipartite graph
*       asize: the number of vertices in the first part of the bipartite graph
* bsize-asize: the number of vertices in the second part
*        0..(asize-1) > A vertices
*        asize..bsize > B vertices
*
* Returns:
*  cover : the actual cover (array)
*  csize : the size of the cover
**************************************************************************/
std::vector<int> MinCover(int* xadj, int* adjncy, int asize, int bsize)
{
    std::vector<int> mate(bsize, -1);
    std::vector<int> flag(bsize);
    std::vector<int> level(bsize);
    std::vector<int> queue(bsize);
    std::vector<int> lst(asize);

    // get cheap matching
    for (int i = 0; i < asize; i++) {
        for (int j = xadj[i]; j < xadj[i + 1]; j++) {
            int v = adjncy[j];
            if (mate[v] == -1) {
                mate[i] = v;
                mate[v] = i;
                break;
            }
        }
    }

    while (1) {
        // initialization
        int fptr = 0; int rptr = 0; int lstptr = 0;
        for (int i = 0; i < bsize; i++) {
            flag[i] = 0;
            level[i] = -1;
        }
        int maxlevel = bsize;

        // insert free nodes into queue
        for (int i = 0; i < asize; i++) {
            if (mate[i] == -1) {
                queue[rptr++] = i;
                level[i] = 0;
            }
        }

        // perform BFS
        while (fptr != rptr) {
            int row = queue[fptr++];
            if (level[row] < maxlevel) {
                flag[row] = 1;
                for (int j = xadj[row]; j < xadj[row+1]; j++) {
                    int col = adjncy[j];
                    if (!flag[col]) {  /* If this column has not been accessed yet */
                        flag[col] = 1;
                        if (mate[col] == -1) { /* Free column node was found */
                            maxlevel = level[row];
                            lst[lstptr++] = col;
                        } else { /* This column node is matched */
                            assert(flag[mate[col]] == 0 && "Something wrong, flag[mate[col]] is not 0");
                            // if (flag[mate[col]]) {
                            //     std::cout << "\nSomething wrong, flag[" << mate[col] << "] is 1" << std::endl;
                            //     exit(1);
                            // }
                            queue[rptr++] = mate[col];
                            level[mate[col]] = level[row] + 1;
                        }
                    }
                }
            } 
        }

        if (lstptr == 0) {
            break; // no augmenting path found
        }

        // perform restricted DFS from the free columns
        for (int i = 0; i < lstptr; i++) {
            MinCover_Augment(xadj, adjncy, lst[i], mate, flag, level, maxlevel);
        }
    }

    std::vector<int> cover = MinCover_Decompose(xadj, adjncy, asize, bsize, mate);

    return cover;
}

// idx1: row indices / indices of the first partition, idx2: column indices / indices of the second partition
// returns bipartite graph of size size1+size2 x size1+size2
// where block (0:size1-1, 0:size1-1) and (size1:size1+size2-1, size1:size1+size2-1) are zero
// and right upper block contains edges from partition 1 to partition 2
// and left lower block contains edges from partition 2 to partition 1
// expects mat to be symmetric, i.e. representing an undirected graph
Eigen::SparseMatrix<int, Eigen::RowMajor> 
getBipartiteGraph(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
			int size1, uint* idx1, 
			int size2, uint* idx2)
{
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(size1*(mat.nonZeros()/mat.rows()));
    std::vector<int> flag(mat.cols(), 0);
    for (int i = 0; i < size2; i++)
        flag[idx2[i]] = 1;

    for (int i = 0; i < size1; i++) {
        for (int j = mat.outerIndexPtr()[idx1[i]]; j < mat.outerIndexPtr()[idx1[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (flag[neighbor] == 1) {
                tripletList.push_back(Eigen::Triplet<int>(i, neighbor, 1));
            }
        }
    }

    Eigen::SparseMatrix<int, Eigen::ColMajor> temp(size1, mat.cols());
    temp.setFromTriplets(tripletList.begin(), tripletList.end());
    temp.makeCompressed();
    tripletList.clear();
    tripletList.reserve((size1+size2)*(temp.nonZeros()/temp.rows()));

    for (int i = 0; i < size2; i++) {
        for (int j = temp.outerIndexPtr()[idx2[i]]; j < temp.outerIndexPtr()[idx2[i] + 1]; j++) {
            int neighbor = temp.innerIndexPtr()[j];
            tripletList.push_back(Eigen::Triplet<int>(neighbor, i + size1, 1));
            tripletList.push_back(Eigen::Triplet<int>(i + size1, neighbor, 1));
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> bip_mat(size1+size2, size1+size2);
    bip_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    bip_mat.makeCompressed();
    return bip_mat;
}

std::vector<int> computeMinSeparator(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
        int size0, int size1, uint* idx)
{
    // get bipartite graph
    Eigen::SparseMatrix<int, Eigen::RowMajor> bip_mat = getBipartiteGraph(mat, size0, idx, size1, idx + size0);
    // compute minimum vertex cover
    std::vector<int> cover = MinCover(bip_mat.outerIndexPtr(), bip_mat.innerIndexPtr(), size0, size0+size1);

    return cover;
}

// pick separator from larger subdomain
std::vector<int> computeSimpleSeparator(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size0, int size1, uint* idx)
{   
    std::vector<int> flag(mat.rows(), 0);
    int i0, max;
    if (size0 < size1) {
        i0 = size0;
        max = size0 + size1;
        for (int i = 0; i < size0; i++) {
            flag[idx[i]] = 1;
        }
    } else {
        i0 = 0;
        max = size0;
        for (int i = size0; i < size0 + size1; i++) {
            flag[idx[i]] = 1;
        }
    }

    std::vector<int> separator; separator.reserve(max - i0);

    for (;i0 < max; i0++) {
        for (int j = mat.outerIndexPtr()[idx[i0]]; j < mat.outerIndexPtr()[idx[i0] + 1]; j++) {
            if (flag[mat.innerIndexPtr()[j]] == 1) {
                separator.push_back(i0);
                break;
            }
        }
    }

    return separator;
}

// pick separator from larger subdomain
std::vector<int> computeSimpleSeparatorNeumann(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size0, int size1, uint* idx, 
    const std::vector<int>& flag_neu, int N_offset)
{   
    std::vector<int> flag(mat.rows(), 0);
    int i0, max;
    if (size0 < size1) {
        i0 = size0;
        max = size0 + size1;
        for (int i = 0; i < size0; i++) {
            flag[idx[i]] = 1;
        }
    } else {
        i0 = 0;
        max = size0;
        for (int i = size0; i < size0 + size1; i++) {
            flag[idx[i]] = 1;
        }
    }

    std::vector<int> separator; separator.reserve(max - i0);

    for (;i0 < max; i0++) {
        int t;
        if (flag_neu[idx[i0]] == 2) { // if ghost node use connectivity of corrsponding neumann node
            t = idx[i0] - N_offset;
        } else {
            t = idx[i0];
        }
        for (int j = mat.outerIndexPtr()[t]; j < mat.outerIndexPtr()[t + 1]; j++) {
            if (flag[mat.innerIndexPtr()[j]] == 1) {
                separator.push_back(i0);
                break;
            }
        }
    }

    return separator;
}

void sortSeparatorToBack(const std::vector<int>& separator, uint* idx, int& size0, int& size1, int& size2, 
    int& size0_after_disection, int& size1_after_disection, int min_size_after_disection)
{
    std::vector<int> part0(size0, 0);
    std::vector<int> part1(size1, 0);
    int counter0 = 0, counter1 = size1;
    for (int i : separator) {
        if (i < size0) {
            part0[i] = 1;
            counter0++;
        } else {
            part1[i - size0] = 1;
            counter1--;
        }
    }

    size0_after_disection = size0 - counter0; 
    size1_after_disection = counter1;

    if (size0_after_disection > min_size_after_disection && size1_after_disection > min_size_after_disection) {
        std::vector<int> part2(counter0 + counter1, 0);
        for (int i = 0; i < counter0; i++)
            part2[i] = 1;

        int size0_tmp = 0, sep0_tmp = 0, size1_tmp = 0, sep1_tmp = 0, dummy0 = 0, dummy1 = 0;
        sortByPartition(part0, idx, size0, size0_tmp, sep0_tmp); // sort part of the separator that is in partition 0 to the back of partition 0
        sortByPartition(part1, idx + size0, size1, size1_tmp, sep1_tmp); // sort part of the separator that is in partition 1 to the back of partition 1
        sortByPartition(part2, idx + size0_tmp, sep0_tmp + size1_tmp, dummy0, dummy1); // sort the separator in partition 0 to the back of the entire partition
        size0 = size0_tmp;
        size1 = size1_tmp;
        size2 = sep0_tmp + sep1_tmp;
    }
}

} // namespace

struct ClusteringDataVel {
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat; // matrix to be partitioned
    pclustergeometry cgv; // positions of the nodes
    std::vector<int> flag_neu; // flags for neumann nodes
    int max_leaf_size; // maximum size of a leaf cluster 
    int N_offset; // offset between Neumann nodes and their corresponding ghost nodes
    int dim; // dimension of the problem
    int min_size_after_disection; // minimum size after disection to continue disection
    bool is_dd_cluster = true; // true if this is a domain decomposition cluster, false if it is a bisection cluster
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel_near = Eigen::SparseMatrix<int, Eigen::RowMajor>(); // velocity nearest neighbor graph
};

struct ClusteringDataCoupledVel : public ClusteringDataVel {
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_grad_div; // pressure vel interaction matrix used for vel coupled clustering
    bool strict = false; // if true stop velocity partitioning if associated pressure cluster is a leaf cluster
};

struct ClusteringDataP {
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div_grad; // pressure matrix to be partitioned
    pclustergeometry cgp; // positions of the pressure nodes
    int max_leaf_size; // maximum size of a leaf cluster 
    int dim; // dimension of the problem
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_schur; // usually mat_grad*mat_div to have sparsity pattern for schur complement to get algebraic clustering
};

std::vector<int> partitionMetisWrapper(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, pclustergeometry, int size, uint* idx)
{
    return partitionVelocityMetis(mat, size, idx);
}

std::vector<int> partitionGeomWrapper(const Eigen::SparseMatrix<int, Eigen::RowMajor>&, pclustergeometry cg, int size, uint* idx)
{
    return partitionVelocityGeom(cg, size, idx);
}

struct Partitioner
{
    std::vector<int>(*partition)(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, pclustergeometry cg, int size, uint* idx);
};

struct Separator
{
    std::vector<int>(*separate)(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int size0, int size1, uint* idx);
};

static pcluster build_dd_cluster_vel(int size, uint* idx, const ClusteringDataVel& data_vel, const Partitioner& part_strategy, const Separator& sep_strategy, bool is_dd_cluster = true)
{
    pcluster c_new;

    if (size > data_vel.max_leaf_size) {
        int size0 = 0, size1 = 0, size2 = 0, size0_after_disection = 0, size1_after_disection = 0;

        {
            std::vector<int> partition = part_strategy.partition(data_vel.mat, data_vel.cgv, size, idx);
            sortVelocity(partition, data_vel.flag_neu, idx, size, size0, size1, data_vel.N_offset);
        }
        if (is_dd_cluster) {
            std::vector<int> separator;
            if (data_vel.N_offset > 0) {
                separator = computeSimpleSeparatorNeumann(data_vel.mat, size0, size1, idx, data_vel.flag_neu, data_vel.N_offset);
            } else {
                separator = sep_strategy.separate(data_vel.mat, size0, size1, idx);
            }
            // separator is only sorted to the back if both remaining partitions after removing the separator would be larger than min_size_after_disection
            sortSeparatorToBack(separator, idx, size0, size1, size2, size0_after_disection, size1_after_disection, data_vel.min_size_after_disection);
        }
        if (size0_after_disection > data_vel.min_size_after_disection && size1_after_disection > data_vel.min_size_after_disection) {
            is_dd_cluster = true;
            if (size0 > 0 && size1 > 0) {
                if (size2 > 0) {
                    c_new = new_cluster(size, idx, 3, data_vel.dim);
                    c_new->son[0] = build_dd_cluster_vel(size0, idx, data_vel, part_strategy, sep_strategy, is_dd_cluster);
                    c_new->son[1] = build_dd_cluster_vel(size1, idx + size0, data_vel, part_strategy, sep_strategy, is_dd_cluster);
                    c_new->son[2] = build_bfs_interface_cluster(data_vel.mat, size2, idx + size0 + size1, data_vel.max_leaf_size, data_vel.dim, 1);
                } else {
                    // no interface, set0 and set1 are disconnected
                    assert(size2 == 0);
                    c_new = new_cluster(size, idx, 2, data_vel.dim);
                    c_new->son[0] = build_dd_cluster_vel(size0, idx, data_vel, part_strategy, sep_strategy, is_dd_cluster);
                    c_new->son[1] = build_dd_cluster_vel(size1, idx + size0, data_vel, part_strategy, sep_strategy, is_dd_cluster);
                }
            } else {
                // size == 0 or size1 == 0
                c_new = new_cluster(size, idx, 0, data_vel.dim);
            }
            c_new->type = 1; // domain cluster
        } else {
            // seperator is (almost) entirety of partition 0 or 1 -> remaining vertices are highly connected
            // therefore no sense in continuing nested disection
            // do geometric bisection instead
            c_new = new_cluster(size, idx, 2, data_vel.dim);
            c_new->type = 3; // bisection cluster
            is_dd_cluster = false;
            c_new->son[0] = build_dd_cluster_vel(size0, idx, data_vel, {&partitionGeomWrapper}, sep_strategy, is_dd_cluster);
            c_new->son[1] = build_dd_cluster_vel(size1, idx + size0, data_vel, {&partitionGeomWrapper}, sep_strategy, is_dd_cluster);
        }
    } else {
        // size < clf
        c_new = new_cluster(size, idx, 0, data_vel.dim);
        if (is_dd_cluster)
            c_new->type = 1; // domain cluster
        else
            c_new->type = 3; // bisecton cluster
    }

    update_cluster(c_new);

    return c_new;
}

static pcluster build_dd_cluster_vel_near(int size, uint* idx, const ClusteringDataVel& data_vel, const Partitioner& part_strategy, const Separator& sep_strategy, 
    int depth, bool is_dd_cluster = true)
{
    if (depth == 0)
        return build_dd_cluster_vel(size, idx, data_vel, part_strategy, sep_strategy, is_dd_cluster);
    
    pcluster c_new;

    if (size > data_vel.max_leaf_size) {
        int size0 = 0, size1 = 0, size2 = 0, size0_after_disection = 0, size1_after_disection = 0;

        {
            std::vector<int> partition = part_strategy.partition(data_vel.mat_vel_near, data_vel.cgv, size, idx);
            sortVelocity(partition, data_vel.flag_neu, idx, size, size0, size1, data_vel.N_offset);
        }
        if (is_dd_cluster) {
            std::vector<int> separator = sep_strategy.separate(data_vel.mat_vel_near, size0, size1, idx);
            // separator is only sorted to the back if both remaining partitions after removing the separator would be larger than min_size_after_disection
            sortSeparatorToBack(separator, idx, size0, size1, size2, size0_after_disection, size1_after_disection, data_vel.min_size_after_disection);
        }
        if (size0_after_disection > data_vel.min_size_after_disection && size1_after_disection > data_vel.min_size_after_disection) {
            is_dd_cluster = true;
            if (size0 > 0 && size1 > 0) {
                if (size2 > 0) {
                    c_new = new_cluster(size, idx, 3, data_vel.dim);
                    c_new->son[0] = build_dd_cluster_vel_near(size0, idx, data_vel, part_strategy, sep_strategy, depth-1, is_dd_cluster);
                    c_new->son[1] = build_dd_cluster_vel_near(size1, idx + size0, data_vel, part_strategy, sep_strategy, depth-1, is_dd_cluster);
                    c_new->son[2] = build_bfs_interface_cluster(data_vel.mat_vel_near, size2, idx + size0 + size1, data_vel.max_leaf_size, data_vel.dim, 1);
                } else {
                    // no interface, set0 and set1 are disconnected
                    assert(size2 == 0);
                    c_new = new_cluster(size, idx, 2, data_vel.dim);
                    c_new->son[0] = build_dd_cluster_vel_near(size0, idx, data_vel, part_strategy, sep_strategy, depth-1, is_dd_cluster);
                    c_new->son[1] = build_dd_cluster_vel_near(size1, idx + size0, data_vel, part_strategy, sep_strategy, depth-1, is_dd_cluster);
                }
            } else {
                // size == 0 or size1 == 0
                c_new = new_cluster(size, idx, 0, data_vel.dim);
            }
            c_new->type = 1; // domain cluster
        } else {
            // seperator is (almost) entirety of partition 0 or 1 -> remaining vertices are highly connected
            // therefore no sense in continuing nested disection
            // do geometric bisection instead
            c_new = new_cluster(size, idx, 2, data_vel.dim);
            c_new->type = 3; // bisection cluster
            is_dd_cluster = false;
            c_new->son[0] = build_dd_cluster_vel(size0, idx, data_vel, {&partitionGeomWrapper}, sep_strategy, is_dd_cluster);
            c_new->son[1] = build_dd_cluster_vel(size1, idx + size0, data_vel, {&partitionGeomWrapper}, sep_strategy, is_dd_cluster);
        }
    } else {
        // size < clf
        c_new = new_cluster(size, idx, 0, data_vel.dim);
        if (is_dd_cluster)
            c_new->type = 1; // domain cluster
        else
            c_new->type = 3; // bisecton cluster
    }

    update_cluster(c_new);

    return c_new;
}

static pcluster
build_adaptive_interface_cluster(pclustergeometry cg, int size, uint * idx,
				 int clf, int dim, int levelint)
{
    pcluster  c;

    uint      size0, size1;
    uint      i, j, direction;
    real      a, m;

    size0 = 0;
    size1 = 0;

    if (size > clf) {
        if (levelint % dim) {
            levelint++;

            update_point_bbox_clustergeometry(cg, size, idx);

            /* compute the direction of partition */
            direction = 0;
            a = cg->hmax[0] - cg->hmin[0];
            for (j = 1; j < cg->dim; j++) {
                m = cg->hmax[j] - cg->hmin[j];
                if (a < m) {
                    a = m;
                    direction = j;
                }
            }

            /* build sons */
            if (a > 0.0) {
                m = (cg->hmax[direction] + cg->hmin[direction]) / 2.0;
                size0 = 0;
                size1 = 0;

                for (i = 0; i < static_cast<uint>(size); i++) {
                    if (cg->x[idx[i]][direction] < m) {
                        j = idx[i];
                        idx[i] = idx[size0];
                        idx[size0] = j;
                        size0++;
                    }
                    else {
                        size1++;
                    }
                }
                if (size0 > 0) {
                    if (size1 > 0) {
                        c = new_cluster(size, idx, 2, cg->dim);

                        c->son[0] = build_adaptive_interface_cluster(cg, size0, idx, clf, dim, levelint);
                        c->son[1] = build_adaptive_interface_cluster(cg, size1, idx + size0, clf,dim, levelint);
                        update_bbox_cluster(c);
                    }
                    else {
                        assert(size0 > 0);
                        assert(size1 == 0);
                        c = new_cluster(size, idx, 1, cg->dim);
                        c->son[0] = build_adaptive_interface_cluster(cg, size, idx, clf, dim, levelint);
                        update_bbox_cluster(c);
                    }
                }
                else {
                    assert(size0 == 0);
                    assert(size1 > 0);
                    c = new_cluster(size, idx, 1, cg->dim);
                    c->son[0] = build_adaptive_interface_cluster(cg, size, idx, clf, dim, levelint);
                    update_bbox_cluster(c);
                }

            }
            else {
                assert(a == 0.0);
                c = new_cluster(size, idx, 0, cg->dim);
                update_support_bbox_cluster(cg, c);
            }
        }

        else {
            levelint++;
            c = new_cluster(size, idx, 1, cg->dim);
            c->son[0] = build_adaptive_interface_cluster(cg, size, idx, clf, dim, levelint);
            update_bbox_cluster(c);
        }


    }

    else {
        /* size <= clf */
        c = new_cluster(size, idx, 0, cg->dim);
        update_support_bbox_cluster(cg, c);
    }

    c->type = 2;
    update_cluster(c);

    return c;
}

static void adjust2vel(std::vector<uint>& sizes, uint* idx, ClusteringDataCoupledVel& data_vel)
{
    std::vector<int> flag(data_vel.mat.cols(), 0);

    for (uint i = 0; i < sizes[0]; i++) {
        uint ii = idx[i];
        for (int k = data_vel.mat.outerIndexPtr()[ii]; k < data_vel.mat.outerIndexPtr()[ii + 1]; k++)
            flag[data_vel.mat.innerIndexPtr()[k]] = 1;
    }

    // identify nonzero entries in block (set0, set1) and move them to the interface (set2)
    for (uint i = sizes[0]; i < sizes[0] + sizes[1]; i++) {
        if (flag[idx[i]] == 1) {
            uint tmp = idx[i];
            idx[i] = idx[sizes[0] + sizes[1] - 1];
            idx[sizes[0] + sizes[1] - 1] = tmp;
            sizes[1] -= 1;
            sizes[2] += 1;
        }
    }

    // this is redundant under the assumption that data_vel.mat is symmetric

    // // reset flag
    // for (uint i = 0; i < flag.size(); i++)
    //     flag[i] = 0;

    // for (uint i = sizes[0]; i < sizes[0] + sizes[1]; i++) {
    //     uint ii = idx[i];
    //     for (int k = data_vel.mat.outerIndexPtr()[ii]; k < data_vel.mat.outerIndexPtr()[ii + 1]; k++)
    //         flag[data_vel.mat.innerIndexPtr()[k]] = 1;
    // }

    // // identify nonzero entries in block (set1, set0) and move them to the interface (set2)
    // for (uint i = 0; i < sizes[0]; i++) {
    //     if (flag[idx[i]] == 1) {
    //         uint tmp = idx[i];
    //         idx[i] = idx[sizes[0] - 1];
    //         idx[sizes[0] - 1] = idx[sizes[0] + sizes[1] - 1];
    //         idx[sizes[0] + sizes[1] - 1] = tmp; 
    //         sizes[0] -= 1;
    //         sizes[2] += 1;
    //     }
    // }

}

pcluster build_dd_coupled_cluster_vel(int size, uint* idx, ClusteringDataCoupledVel& data_vel, pcluster associate)
{
    pcluster c;

    if (associate->sons > 1) {
        assert(associate->sons == 2);

        std::vector<uint> sizes(3, 0);

        {
            std::vector<int> flag(data_vel.mat.cols(), 0);

            // Set flag for nodes connected to the first cluster
            pcluster son = associate->son[1];
            for (uint i = 0; i < son->size; i++) {
                uint ii = son->idx[i];
                for (int k = data_vel.mat_grad_div.outerIndexPtr()[ii]; k < data_vel.mat_grad_div.outerIndexPtr()[ii + 1]; k++)
                    flag[data_vel.mat_grad_div.innerIndexPtr()[k]] = 1;
            }

            for (int i = 0; i < size; i++) {
                if (flag[idx[i]] == 0) {
                    uint tmp = idx[i];
                    idx[i] = idx[sizes[0]];
                    idx[sizes[0]] = tmp;
                    sizes[0] += 1;
                }
                else {
                    sizes[1] += 1;
                }
            }

            // Reset flag
            for (uint i = 0; i < flag.size(); i++)
                flag[i] = 0;

            son = associate->son[0];
            for (uint i = 0; i < son->size; i++) {
                uint ii = son->idx[i];
                for (int k = data_vel.mat_grad_div.outerIndexPtr()[ii]; k < data_vel.mat_grad_div.outerIndexPtr()[ii + 1]; k++)
                    flag[data_vel.mat_grad_div.innerIndexPtr()[k]] = 1;
            }

            uint j = sizes[0];
            while (j < sizes[0] + sizes[1]) {
                if (flag[idx[j]] == 1) {
                    uint tmp = idx[j];
                    idx[j] = idx[size - sizes[2] - 1];
                    idx[size - sizes[2] - 1] = tmp;
                    sizes[1] -= 1;
                    sizes[2] += 1;
                }
                else {
                    j++;
                }
            }
        } // order idx according to associated cluster

        // this makes sure that the velocity clustering leads to arrow head structure (blocks (set0, set1) and (set1, set0) are zero)
        // for the FEM discretization of the Oseen equations with Taylor-Hood elements this wasn't necessary, because you can prove that 
        // this structure is already given by construction, but for general RBF-FD discretizations this is not the case
        adjust2vel(sizes, idx, data_vel);

        std::vector<pcluster> sons;
        uint offset = 0;

        if (sizes[0] > 0) {
            pcluster son = build_dd_coupled_cluster_vel(sizes[0], idx + offset, data_vel, associate->son[0]);

            sons.push_back(son);
            offset += sizes[0];
        }
        if (sizes[1] > 0) {
            pcluster son = build_dd_coupled_cluster_vel(sizes[1], idx + offset, data_vel, associate->son[1]);

            sons.push_back(son);
            offset += sizes[1];
        }
        if (sizes[2] > 0) {
            pcluster son = build_adaptive_interface_cluster(data_vel.cgv, sizes[2], idx + offset, data_vel.max_leaf_size, data_vel.dim, 1);

            sons.push_back(son);
            offset += sizes[2];
        }

        assert(offset == static_cast<uint>(size));

        c = new_cluster(size, idx, sons.size(), data_vel.dim);
        for (uint i = 0; i < sons.size(); i++)
        {
            c->son[i] = sons[i];
        }
        update_bbox_cluster(c);
    }
    else if (associate->sons == 1) {
        // cluster_p has only one son
        c = new_cluster(size, idx, 1, data_vel.dim);

        c->son[0] = build_dd_coupled_cluster_vel(size, idx, data_vel, associate);

        update_bbox_cluster(c);
    }
    else if (data_vel.strict == false && size > data_vel.max_leaf_size) {
        c = build_dd_cluster_vel(size, idx, data_vel, {&partitionGeomWrapper}, {&computeSimpleSeparator}, true);
    }
    else {
        c = new_cluster(size, idx, 0, data_vel.dim);

        update_support_bbox_cluster(data_vel.cgv, c);
    }

    update_cluster(c);

    c->type = 1;
    c->associated = associate;
    associate->associated = c;

    return c;
}

static std::vector<int> getPressureSeparator(Eigen::SparseMatrix<int, Eigen::RowMajor>& matp,
    int size0, int size1, uint* idx, 
    int size0p, int size1p, uint* idxp)
{
    std::vector<int> flag0(matp.cols(), 0);
    std::vector<int> flag1(matp.cols(), 0);
    for (int i = 0; i < size0; i++)
        flag0[idx[i]] = 1;
    
    for (int i = size0; i < size0 + size1; i++)
        flag1[idx[i]] = 1;

    std::vector<int> separator; separator.reserve(size0p + size1p);

    for (int i = 0; i < size0p; i++) {
        for (int j = matp.outerIndexPtr()[idxp[i]]; j < matp.outerIndexPtr()[idxp[i] + 1]; j++) {
            if (flag1[matp.innerIndexPtr()[j]] == 1) {
                separator.push_back(i);
                break;
            }
        }
    }

    for (int i = size0p; i < size0p + size1p; i++) {
        for (int j = matp.outerIndexPtr()[idxp[i]]; j < matp.outerIndexPtr()[idxp[i] + 1]; j++) {
            if (flag0[matp.innerIndexPtr()[j]] == 1) {
                separator.push_back(i);
                break;
            }
        }
    }

    return separator;
}

pcluster build_coupled_cluster_p(int sizep, uint* idxp, ClusteringDataP& data_p, pcluster c_vel)
{
    int dim = data_p.dim;
    int size0p = 0, size1p = 0;

    pcluster c_new;

    if (sizep > data_p.max_leaf_size) {
        if (c_vel->sons >= 2) {
            int size0 = c_vel->son[0]->size;
            int size1 = c_vel->son[1]->size;
            uint* idx = c_vel->idx;
            partitionAndSortPressure(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, sizep, idxp);

            if (size0p > 0 && size1p > 0) {
                c_new = new_cluster(sizep, idxp, 2, dim);
                c_new->son[0] = build_coupled_cluster_p(size0p, idxp, data_p, c_vel->son[0]);
                c_new->son[1] = build_coupled_cluster_p(size1p, idxp+size0p, data_p, c_vel->son[1]);
                update_bbox_cluster(c_new);
            } else {
                // no reasonable algebraic partitioning possible, continue with geometric bisection
                c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
                update_support_bbox_cluster(data_p.cgp, c_new);
            }
        } else {
            // no more velocity clusters, switch to geometric bisection
            c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
            update_support_bbox_cluster(data_p.cgp, c_new);
        }
        
    } else {
        // sizep < clfp or no velocity partitioning
        c_new = new_cluster(sizep, idxp, 0, dim);
        update_support_bbox_cluster(data_p.cgp, c_new);
    }

    update_cluster(c_new);
    return c_new;
}

static pcluster build_coupled_cluster_p_with_interface(int sizep, uint* idxp, ClusteringDataP& data_p, pcluster c_vel)
{
    int dim = data_p.dim;
    int size0p = 0, size1p = 0;

    pcluster c_new;

    if (sizep > data_p.max_leaf_size) {
        if (c_vel->sons >= 2) {
            int size2p = 0;
            {
                int size0p_after_disection = 0, size1p_after_disection = 0;
                int size0 = c_vel->son[0]->size;
                int size1 = c_vel->son[1]->size;
                uint* idx = c_vel->idx;
                partitionAndSortPressure(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, sizep, idxp);
                std::vector<int> separator = getPressureSeparator(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, idxp);
                // 10 is min size after dissection, i.e. if size0 or size1 < 10, then we just do bissection and don't work with the separator
                sortSeparatorToBack(separator, idxp, size0p, size1p, size2p, size0p_after_disection, size1p_after_disection, 10);
            }

            if (size0p > 0 && size1p > 0) {
                if (size2p > 0) {
                    // a separator was found
                    c_new = new_cluster(sizep, idxp, 3, dim);
                    c_new->son[0] = build_coupled_cluster_p_with_interface(size0p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_with_interface(size1p, idxp+size0p, data_p, c_vel->son[1]);
                    c_new->son[2] = build_adaptive_interface_cluster(data_p.cgp, size2p, idxp + size0p + size1p, data_p.max_leaf_size, dim, 1);
                } else {
                    // no separator, just bisection
                    c_new = new_cluster(sizep, idxp, 2, dim);
                    c_new->son[0] = build_coupled_cluster_p_with_interface(size0p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_with_interface(size1p, idxp+size0p, data_p, c_vel->son[1]);
                }
                
                update_bbox_cluster(c_new);
            } else {
                // no reasonable algebraic partitioning possible, try geometric bisection
                c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
                update_support_bbox_cluster(data_p.cgp, c_new);
            }
        } else {
            // no more velocity clusters, switch to geometric bisection
            c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
            update_support_bbox_cluster(data_p.cgp, c_new);
        }
        
    } else {
        // sizep < clfp or no velocity partitioning
        c_new = new_cluster(sizep, idxp, 0, dim);
        update_support_bbox_cluster(data_p.cgp, c_new);
    }

    c_new->type = 1; // domain cluster
    update_cluster(c_new);
    return c_new;
}

static pcluster build_coupled_cluster_p_one_zero_block(int sizep, uint* idxp, ClusteringDataP& data_p, pcluster c_vel)
{
    int dim = data_p.dim;
    int size0p = 0, size1p = 0;

    pcluster c_new;

    if (sizep > data_p.max_leaf_size) {
        if (c_vel->sons >= 2) {
            int size2p = 0;
            {
                int size0p_after_disection = 0, size1p_after_disection = 0;
                int size0 = c_vel->son[0]->size;
                int size1 = c_vel->son[1]->size;
                uint* idx = c_vel->idx;
                partitionAndSortPressure(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, sizep, idxp);
                std::vector<int> separator = getPressureSeparator(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, idxp);
                sortSeparatorToBack(separator, idxp, size0p, size1p, size2p, size0p_after_disection, size1p_after_disection, -1);
                if (size0p < size1p) {
                    // leave idx0p as it is and sort idx1p to the back and separator idxs inbetween
                    std::vector<int> partition(size1p+size2p, 0);
                    for (int i = 0; i < size1p; i++)
                        partition[i] = 1;
                    int dummy0 = 0, dummy1 = 0;
                    sortByPartition(partition, idxp + size0p, size1p + size2p, dummy0, dummy1);
                    assert(dummy0 == size2p);
                    assert(dummy1 == size1p);
                }
            }
            // choose where the separator goes based on sizes to keep the clusters more balanced
            if (size0p >= size1p) {
                if (size0p > 0 && size1p+size2p > 0) {
                    c_new = new_cluster(sizep, idxp, 2, dim);
                    c_new->son[0] = build_coupled_cluster_p_one_zero_block(size0p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_one_zero_block(size1p+size2p, idxp+size0p, data_p, c_vel->son[1]);
                    
                    update_bbox_cluster(c_new);
                } else {
                    // no reasonable algebraic partitioning possible, try geometric bisection
                    c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
                    update_support_bbox_cluster(data_p.cgp, c_new);
                }
            } else {
                if (size0p+size2p > 0 && size1p > 0) {
                    c_new = new_cluster(sizep, idxp, 2, dim);
                    c_new->son[0] = build_coupled_cluster_p_one_zero_block(size0p+size2p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_one_zero_block(size1p, idxp+size0p+size2p, data_p, c_vel->son[1]);
                    
                    update_bbox_cluster(c_new);
                } else {
                    // no reasonable algebraic partitioning possible, try geometric bisection
                    c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
                    update_support_bbox_cluster(data_p.cgp, c_new);
                }
            }
        } else {
            // no more velocity clusters, switch to geometric bisection
            c_new = build_cluster(data_p.cgp, sizep, idxp, data_p.max_leaf_size, H2_ADAPTIVE);
            update_support_bbox_cluster(data_p.cgp, c_new);
        }
        
    } else {
        // sizep < clfp or no velocity partitioning
        c_new = new_cluster(sizep, idxp, 0, dim);
        update_support_bbox_cluster(data_p.cgp, c_new);
    }

    c_new->type = 1; // domain cluster
    update_cluster(c_new);
    return c_new;
}

static pcluster build_coupled_cluster_p_algebraic(int sizep, uint* idxp, ClusteringDataP& data_p, pcluster c_vel)
{
    int dim = data_p.dim;
    int size0p = 0, size1p = 0;

    pcluster c_new;

    if (sizep > data_p.max_leaf_size) {
        if (c_vel->sons >= 2) {
            int size0 = c_vel->son[0]->size;
            int size1 = c_vel->son[1]->size;
            uint* idx = c_vel->idx;
            partitionAndSortPressure(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, sizep, idxp);

            if (size0p > 0 && size1p > 0) {
                c_new = new_cluster(sizep, idxp, 2, dim);
                c_new->son[0] = build_coupled_cluster_p_algebraic(size0p, idxp, data_p, c_vel->son[0]);
                c_new->son[1] = build_coupled_cluster_p_algebraic(size1p, idxp+size0p, data_p, c_vel->son[1]);
            } else {
                c_new = new_cluster(sizep, idxp, 0, dim);
            }
        } else {
            c_new = new_cluster(sizep, idxp, 0, dim);
        }
    } else {
        // sizep < clfp or no velocity partitioning
        c_new = new_cluster(sizep, idxp, 0, dim);
    }

    update_cluster(c_new);
    return c_new;
}

// this needs an algebraic version of build_adaptive_interface_cluster to actually work fully algebraically
static pcluster build_coupled_cluster_p_with_interface_algebraic(int sizep, uint* idxp, ClusteringDataP& data_p, pcluster c_vel)
{
    int dim = data_p.dim;
    int size0p = 0, size1p = 0;

    pcluster c_new;

    if (sizep > data_p.max_leaf_size) {
        if (c_vel->sons >= 2) {
            int size2p = 0;
            {
                int size0p_after_disection = 0, size1p_after_disection = 0;
                int size0 = c_vel->son[0]->size;
                int size1 = c_vel->son[1]->size;
                uint* idx = c_vel->idx;
                partitionAndSortPressure(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, sizep, idxp);
                std::vector<int> separator = getPressureSeparator(data_p.mat_div_grad, size0, size1, idx, size0p, size1p, idxp);
                // 10 is min size after dissection, i.e. if size0 or size1 < 10, then we just do bissection and don't work with the separator
                sortSeparatorToBack(separator, idxp, size0p, size1p, size2p, size0p_after_disection, size1p_after_disection, 10);
            }

            if (size0p > 0 && size1p > 0) {
                if (size2p > 0) {
                    // a separator was found
                    c_new = new_cluster(sizep, idxp, 3, dim);
                    c_new->son[0] = build_coupled_cluster_p_with_interface_algebraic(size0p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_with_interface_algebraic(size1p, idxp+size0p, data_p, c_vel->son[1]);
                    c_new->son[2] = build_adaptive_interface_cluster(data_p.cgp, size2p, idxp + size0p + size1p, data_p.max_leaf_size, dim, 1);
                } else {
                    // no separator, just bisection
                    c_new = new_cluster(sizep, idxp, 2, dim);
                    c_new->son[0] = build_coupled_cluster_p_with_interface_algebraic(size0p, idxp, data_p, c_vel->son[0]);
                    c_new->son[1] = build_coupled_cluster_p_with_interface_algebraic(size1p, idxp+size0p, data_p, c_vel->son[1]);
                }
            } else {
                c_new = new_cluster(sizep, idxp, 0, dim);
            }
        } else {
            c_new = new_cluster(sizep, idxp, 0, dim);
        }
        
    } else {
        // sizep < clfp or no velocity partitioning
        c_new = new_cluster(sizep, idxp, 0, dim);
    }

    c_new->type = 1; // domain cluster
    update_cluster(c_new);
    return c_new;
}

static pcluster build_bisection_cluster_metis(int size, uint* idx, const ClusteringDataP& data_p)
{
    pcluster c_new;

    if (size > data_p.max_leaf_size) {
        int size0 = 0, size1 = 0;

        {
            std::vector<int> partition = partitionMetisWrapper(data_p.mat_schur, data_p.cgp, size, idx);
            sortByPartition(partition, idx, size, size0, size1);
        }

        c_new = new_cluster(size, idx, 2, data_p.dim);
        c_new->son[0] = build_bisection_cluster_metis(size0, idx, data_p);
        c_new->son[1] = build_bisection_cluster_metis(size1, idx + size0, data_p);
        update_bbox_cluster(c_new);
    } else {
        // size < clf
        c_new = new_cluster(size, idx, 0, data_p.dim);
        update_support_bbox_cluster(data_p.cgp, c_new);
    }

    update_cluster(c_new);

    return c_new;
}

static pcluster build_bisection_cluster_metis(int size, uint* idx, const ClusteringDataVel& data_vel)
{
    pcluster c_new;

    if (size > data_vel.max_leaf_size) {
        int size0 = 0, size1 = 0;

        {
            std::vector<int> partition = partitionMetisWrapper(data_vel.mat, data_vel.cgv, size, idx);
            sortByPartition(partition, idx, size, size0, size1);
        }

        c_new = new_cluster(size, idx, 2, data_vel.dim);
        c_new->son[0] = build_bisection_cluster_metis(size0, idx, data_vel);
        c_new->son[1] = build_bisection_cluster_metis(size1, idx + size0, data_vel);
    } else {
        // size < clf
        c_new = new_cluster(size, idx, 0, data_vel.dim);
    }

    update_cluster(c_new);

    return c_new;
}

std::vector<int> sortByNPartsPartition(const std::vector<int>& partition, uint* idx, int size, int n_parts)
{
    assert(partition.size() == static_cast<size_t>(size));
    // get sizes of subdomains
    std::vector<int> subdomain_sizes(n_parts, 0);
    for (int i : partition) {
        subdomain_sizes[i]++;
    }
    std::vector<int> subdomain_pointers(n_parts, 0); // point to the beginning of each subdomain
    for (int i = 1; i < n_parts; i++) {
        subdomain_pointers[i] = subdomain_pointers[i - 1] + subdomain_sizes[i - 1];
    }

    std::vector<int> temp_idx(size);
    // sort according to partition (partition 0 first then partition 1, etc.)
    for (int i = 0; i < size; i++) {
        temp_idx[subdomain_pointers[partition[i]]] = idx[i];
        subdomain_pointers[partition[i]]++;
    }
    // copy back to idx
    for (int i = 0; i < size; i++) {
        idx[i] = temp_idx[i];
    }

    return subdomain_sizes;
}

// this version performs idle steps if the size is already small enough for the current depth
static pcluster build_bisection_cluster_metis(int size, uint* idx, const ClusteringDataVel& data_vel, int initial_size, int target_depth, int current_depth)
{
    pcluster c_new;

    double factor = std::pow((static_cast<double>(data_vel.max_leaf_size) / static_cast<double>(initial_size)), (1.0/static_cast<double>(target_depth)));

    if (size > data_vel.max_leaf_size) {
        if (size < initial_size * std::pow(factor, current_depth)) {
            std::cout << 1 << ", ";
            c_new = new_cluster(size, idx, 1, data_vel.dim);
            c_new = build_bisection_cluster_metis(size, idx, data_vel, initial_size, target_depth, current_depth + 1);
        } else {
            int n_parts = std::max(std::ceil(static_cast<double>(size) / (initial_size * std::pow(factor, current_depth))) , 2.0);
            std::cout << n_parts << ", ";

            std::vector<int> partition = partitionVelocityMetis(data_vel.mat, size, idx, n_parts);
            std::vector<int> subdomain_sizes = sortByNPartsPartition(partition, idx, size, n_parts);

            c_new = new_cluster(size, idx, n_parts, data_vel.dim);
            int offset = 0;
            for (int i = 0; i < n_parts; i++) {
                c_new->son[i] = build_bisection_cluster_metis(subdomain_sizes[i], idx + offset, data_vel, initial_size, target_depth, current_depth + 1);
                offset += subdomain_sizes[i];
            }
        }
    } else {
        // size < clf
        c_new = new_cluster(size, idx, 0, data_vel.dim);
    }

    update_cluster(c_new);

    return c_new;
}

// test if neumann and ghost nodes ended up in the same cluster
static void test_neumann_vel_cluster(pcluster rootv, std::vector<int>& flag_neu, int N_neu)
{
    for (uint i = 0; i < rootv->size; i++) {
        if (flag_neu[rootv->idx[i]] == 1) {
            // neumann node
            bool found =  false;
            for (uint j = 0; j < rootv->size; j++) {
                if (rootv->idx[j] == rootv->idx[i] + N_neu) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "ghost node corresponding to neumann node " << rootv->idx[i] << " is not in the same cluster" << std::endl;
            }
        }
        if (flag_neu[rootv->idx[i]] == 2) {
            // neumann node
            bool found =  false;
            for (uint j = 0; j < rootv->size; j++) {
                if (rootv->idx[j] == rootv->idx[i] - N_neu) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "neumann node corresponding to ghost node " << rootv->idx[i] << " is not in the same cluster" << std::endl;
            }
        }
    }
    if (rootv->sons > 1) {
        uint n = std::min(rootv->sons, (uint)2);
        for (uint i = 0; i < n; i++) {
            test_neumann_vel_cluster(rootv->son[i], flag_neu, N_neu);
        }
    }
}

clusterStorage getBlackboxClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    int N_u, int N_p, int dim, ClusteringOptions cluster_opt)
{
    // cut out velocity block from the matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = mat.block(0, 0, N_u, N_u).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 0, N_u, N_u).transpose().cast<int>());
    // mat_p = div block of matrix + tranpose of grad block of matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div_grad = mat.block(3*N_u, 0, N_p, N_u).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 3*N_u, N_u, N_p).transpose().cast<int>());

    // choose partitioning and separation strategy
    Partitioner part_strategy{&partitionMetisWrapper};
    Separator sep_strategy;

    if (cluster_opt.separator_type == SeparatorType::SIMPLE)
        sep_strategy.separate = &computeSimpleSeparator;
    else if (cluster_opt.separator_type == SeparatorType::MINIMUM)
        sep_strategy.separate = &computeMinSeparator;

    uint* idx = new uint[N_u]; // this will be deleted when the velocity cluster is deleted
    for (int i = 0; i < N_u; i++)
        idx[i] = i;

    uint* idxp = new uint[N_p+1];
    for (int i = 0; i < N_p+1; i++)
        idxp[i] = i;

    pcluster rootv;
    pcluster rootp = new_cluster(N_p+1, idxp, 2, dim); // cluster containing pressure nodes and constraint
    rootp->son[1] = new_cluster(1, idxp+(N_p), 0, dim); // cluster containing only the pressure constraint;

    ClusteringDataVel data_vel = {
        mat_vel,
        nullptr,
        std::vector<int>(),
        cluster_opt.max_leaf_size_vel,
        0,
        dim,
        cluster_opt.min_size_after_disection
    };

    ClusteringDataP data_p = {
        mat_div_grad,
        nullptr,
        cluster_opt.max_leaf_size_p,
        dim,
        Eigen::SparseMatrix<int, Eigen::RowMajor>()
    };

    rootv = build_dd_cluster_vel(N_u, idx, data_vel, part_strategy, sep_strategy);

    if (cluster_opt.pressure_cluster_type == PressureClusterType::COUPLED_WITH_INTERFACE)
        std::cout << "coupled pressure clustering with interface not supported for blackbox clustering yet" << std::endl;
        // rootp->son[0] = build_coupled_cluster_p_with_interface_algebraic(N_p, idxp, data_p, rootv);
    else if (cluster_opt.pressure_cluster_type == PressureClusterType::COUPLED_NO_INTERFACE)
        rootp->son[0] = build_coupled_cluster_p_algebraic(N_p, idxp, data_p, rootv);
    else
        std::cout << "Blackbox clustering only supports coupled pressure clustering" << std::endl;

    clusterStorage root = {rootv, rootp};

    return root;
}

clusterStorage getClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, int cg_support_size)
{
    int N = dc.N_ui + dc.N_uneu;

    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = mat.block(0, 0, N, N).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 0, N, N).transpose().cast<int>());
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel_near = Eigen::SparseMatrix<int, Eigen::RowMajor>();
    
    if (cluster_opt.connectivity_degree > 0 && cluster_opt.max_depth_near > -1)
        mat_vel_near = createNearestNeighborMatrixVelocity(dc, cluster_opt.connectivity_degree);

    // mat_p = div block of matrix + tranpose of grad block of matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div_grad = mat.block(3*N, 0, dc.N_p, N).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 3*N, N, dc.N_p).transpose().cast<int>());

    std::vector<int> flag_neu(N, 0);
    for (int i : dc.idxs_neu) {
        flag_neu[i-dc.N_ub] = 1; // neumann node
    }
    for (int i : dc.idxs_u_ghost_global) {
        flag_neu[i-dc.N_ub] = 2; // ghost node
    }
    int N_offset = 0;
    if (dc.N_uneu)
        N_offset = dc.idxs_u_ghost_global[0] - dc.idxs_neu[0]; // offset between corresponding neumann and ghost nodes

    // choose partitioning and separation strategy

    Partitioner part_strategy;
    Separator sep_strategy;

    if (cluster_opt.partition_type == PartitionType::METIS)
        part_strategy.partition = &partitionMetisWrapper;
    else if (cluster_opt.partition_type == PartitionType::GEOM)
        part_strategy.partition = &partitionGeomWrapper;

    if (cluster_opt.separator_type == SeparatorType::SIMPLE)
        sep_strategy.separate = &computeSimpleSeparator;
    else if (cluster_opt.separator_type == SeparatorType::MINIMUM)
        sep_strategy.separate = &computeMinSeparator;

    uint* idx = new uint[N]; // this will be deleted when the velocity cluster is deleted
    for (int i = 0; i < N; i++)
        idx[i] = i;

    uint* idxp = new uint[dc.N_p+1];
    for (int i = 0; i < dc.N_p+1; i++)
        idxp[i] = i;

    // pclustergeometry cgp = build_clustergeometry_medusa(dc.d_p, dc.d_p.all());
    // uses supports/stencils to build geometry -> leads to stricter strong admissibility condition
    pclustergeometry cgp = build_clustergeometry_medusa_supports(dc.d_div, mm::Range<int>::seq(dc.d_u.size(), dc.d_div.size()), cg_support_size);
    pclustergeometry cgv = build_clustergeometry_medusa_supports(dc.d_u, mm::Range<int>::seq(dc.N_ub, dc.d_u.size()), cg_support_size);

    // compute clusterings
    pcluster rootv;
    pcluster rootp = new_cluster(dc.N_p+1, idxp, 2, dc.d_p.dim); // cluster containing pressure nodes and constraint
    rootp->son[1] = new_cluster(1, idxp+(dc.N_p), 0, dc.d_p.dim); // cluster containing only the pressure constraint;
    if (cluster_opt.velocity_cluster_type == VelocityClusterType::STANDARD_DD) {
        ClusteringDataVel data_vel = {
            mat_vel,
            cgv,
            flag_neu,
            cluster_opt.max_leaf_size_vel,
            N_offset,
            dc.d_u.dim,
            cluster_opt.min_size_after_disection,
            true,
            mat_vel_near,
        };

        ClusteringDataP data_p = {
            mat_div_grad,
            cgp,
            cluster_opt.max_leaf_size_p,
            dc.d_p.dim,
            Eigen::SparseMatrix<int, Eigen::RowMajor>()
        };

        if (cluster_opt.connectivity_degree > 0 && cluster_opt.max_depth_near > -1)
            rootv = build_dd_cluster_vel_near(N, idx, data_vel, part_strategy, sep_strategy, cluster_opt.max_depth_near);
        else
            rootv = build_dd_cluster_vel(N, idx, data_vel, part_strategy, sep_strategy);

        switch (cluster_opt.pressure_cluster_type)
        {
        case PressureClusterType::COUPLED_WITH_INTERFACE:
            rootp->son[0] = build_coupled_cluster_p_with_interface(dc.N_p, idxp, data_p, rootv);
            break;
        case PressureClusterType::COUPLED_NO_INTERFACE:
            rootp->son[0] = build_coupled_cluster_p(dc.N_p, idxp, data_p, rootv);
            break;
        case PressureClusterType::UNCOUPLED_GEOM:
            rootp->son[0] = build_cluster(cgp, dc.d_p.size(), idxp, cluster_opt.max_leaf_size_p, H2_ADAPTIVE);
            break;
        case PressureClusterType::COUPLED_ONE_ZERO_BLOCK:
            rootp->son[0] = build_coupled_cluster_p_one_zero_block(dc.N_p, idxp, data_p, rootv);
            break;
        case PressureClusterType::UNCOUPLED_METIS:
            {
            Eigen::SparseMatrix<int, Eigen::RowMajor> mat_schur = mat.block(3*N, 0, dc.N_p, N).cast<int>()
                    * mat.block(0, 3*N, N, dc.N_p).cast<int>();
            ClusteringDataP data_p = {
                Eigen::SparseMatrix<int, Eigen::RowMajor>(),
                cgp,
                cluster_opt.max_leaf_size_p,
                dc.d_p.dim,
                mat_schur
            };
            rootp->son[0] = build_bisection_cluster_metis(dc.N_p, idxp, data_p);
            }
            break;
        default:
            break;
        }
    }

    if (cluster_opt.velocity_cluster_type == VelocityClusterType::COUPLED_DD) {
        ClusteringDataCoupledVel data_vel_coupled;
        data_vel_coupled.mat = mat_vel;
        data_vel_coupled.mat_grad_div = mat_div_grad;
        data_vel_coupled.cgv = cgv;
        data_vel_coupled.flag_neu = flag_neu;
        data_vel_coupled.max_leaf_size = cluster_opt.max_leaf_size_vel;
        data_vel_coupled.N_offset = N_offset;
        data_vel_coupled.dim = dc.d_u.dim;
        data_vel_coupled.min_size_after_disection = cluster_opt.min_size_after_disection;
        data_vel_coupled.strict = false;
        switch (cluster_opt.pressure_cluster_type)
        {
        case PressureClusterType::UNCOUPLED_GEOM:
            rootp->son[0] = build_cluster(cgp, dc.d_p.size(), idxp, cluster_opt.max_leaf_size_p, H2_ADAPTIVE);
            break;

        case PressureClusterType::UNCOUPLED_METIS:
            {
            Eigen::SparseMatrix<int, Eigen::RowMajor> mat_schur = mat.block(3*N, 0, dc.N_p, N).cast<int>()
                    * mat.block(0, 3*N, N, dc.N_p).cast<int>();
            ClusteringDataP data_p = {
                Eigen::SparseMatrix<int, Eigen::RowMajor>(),
                cgp,
                cluster_opt.max_leaf_size_p,
                dc.d_p.dim,
                mat_schur
            };
            rootp->son[0] = build_bisection_cluster_metis(dc.N_p, idxp, data_p);
            }
            break;

        default:
            break;
        }
        rootv = build_dd_coupled_cluster_vel(N, idx, data_vel_coupled, rootp->son[0]);
    }

    // test_neumann_vel_cluster(rootv, flag_neu, N_offset);
    
    clusterStorage root = {rootv, rootp};

    return root;
}

clusterStorage getClusteringGeom(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
    const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, int cg_support_size)
{
    int N = dc.N_ui + dc.N_uneu;

    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_vel = mat.block(0, 0, N, N)
                    + static_cast<Eigen::SparseMatrix<double, Eigen::RowMajor>>(mat.block(0, 0, N, N).transpose());

    uint* idx = new uint[N]; // this will be deleted when the velocity cluster is deleted
    for (int i = 0; i < N; i++)
        idx[i] = i;

    uint* idxp = new uint[dc.N_p+1];
    for (int i = 0; i < dc.N_p+1; i++)
        idxp[i] = i;

    // pclustergeometry cgp = build_clustergeometry_medusa(dc.d_p, dc.d_p.all());
    // uses supports/stencils to build geometry -> leads to stricter strong admissibility condition
    pclustergeometry cgp = build_clustergeometry_medusa_supports(dc.d_div, mm::Range<int>::seq(dc.d_u.size(), dc.d_div.size()), cg_support_size);
    pclustergeometry cgv = build_clustergeometry_medusa_supports(dc.d_u, mm::Range<int>::seq(dc.N_ub, dc.d_u.size()), cg_support_size);

    std::vector<uint> flag(N,0);

    psparsematrix sp = matMM2H(mat_vel);

    // compute clusterings
    pcluster rootv = build_adaptive_dd_cluster(cgv, N, idx,
			  cluster_opt.max_leaf_size_vel, sp, dc.d_u.dim, flag.data());
    pcluster rootp = new_cluster(dc.N_p+1, idxp, 2, dc.d_p.dim); // cluster containing pressure nodes and constraint
    rootp->son[1] = new_cluster(1, idxp+(dc.N_p), 0, dc.d_p.dim); // cluster containing only the pressure constraint;
    rootp->son[0] = build_cluster(cgp, dc.d_p.size(), idxp, cluster_opt.max_leaf_size_p, H2_ADAPTIVE);

    clusterStorage root = {rootv, rootp};

    return root;
}

pcluster getCO_DD_VelClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_vel_d, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_div_d,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat_grad_d,
    const mm::OseenDiscretizationBetter& dc, ClusteringOptions cluster_opt, pcluster rootp_given, int cg_support_size)
{
    int N = dc.N_ui + dc.N_uneu;

    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = mat_vel_d.cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat_vel_d.transpose().cast<int>());
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div_grad = (mat_div_d.cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat_grad_d.transpose().cast<int>())).block(0, 0, dc.N_p, N);

    std::vector<int> flag_neu(N, 0);
    for (int i : dc.idxs_neu) {
        flag_neu[i-dc.N_ub] = 1; // neumann node
    }
    for (int i : dc.idxs_u_ghost_global) {
        flag_neu[i-dc.N_ub] = 2; // ghost node
    }
    int N_offset = 0;
    if (dc.N_uneu)
        N_offset = dc.idxs_u_ghost_global[0] - dc.idxs_neu[0]; // offset between corresponding neumann and ghost nodes

    uint* idx = new uint[N]; // this will be deleted when the velocity cluster is deleted
    for (int i = 0; i < N; i++)
        idx[i] = i;

    uint* idxp = new uint[dc.N_p+1];
    for (int i = 0; i < dc.N_p+1; i++)
        idxp[i] = i;

    // uses supports/stencils to build geometry -> leads to stricter strong admissibility condition
    pclustergeometry cgv = build_clustergeometry_medusa_supports(dc.d_u, mm::Range<int>::seq(dc.N_ub, dc.d_u.size()), cg_support_size);

    // compute clusterings
    pcluster rootv;

    ClusteringDataCoupledVel data_vel_coupled;
    data_vel_coupled.mat = mat_vel;
    data_vel_coupled.mat_grad_div = mat_div_grad;
    data_vel_coupled.cgv = cgv;
    data_vel_coupled.flag_neu = flag_neu;
    data_vel_coupled.max_leaf_size = cluster_opt.max_leaf_size_vel;
    data_vel_coupled.N_offset = N_offset;
    data_vel_coupled.dim = dc.d_u.dim;
    data_vel_coupled.min_size_after_disection = cluster_opt.min_size_after_disection;
    data_vel_coupled.strict = false;
    rootv = build_dd_coupled_cluster_vel(N, idx, data_vel_coupled, rootp_given->son[0]);

    return rootv;
}

struct SubdomainIdx
{
    std::vector<int> idx_interior;
    std::vector<int> idx_separator;
};

SubdomainIdx getSubdomainIdx(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, const std::vector<int>& idx,
    int start, int size)
{
    // mark all indices that are NOT in the current subdomain
    std::vector<int> flag(mat.rows(), 1);
    for (int i = start; i < start + size; i++) {
        flag[idx[i]] = 0;
    }

    std::vector<int> idx_interior;
    std::vector<int> idx_separator;
    idx_interior.reserve(size);
    idx_separator.reserve(size);

    bool is_separator = false;

    for (int i = start; i < start + size; i++) {
        for (int j = mat.outerIndexPtr()[idx[i]]; j < mat.outerIndexPtr()[idx[i] + 1]; j++) {
            if (flag[mat.innerIndexPtr()[j]] == 1) {
                idx_separator.push_back(idx[i]);
                is_separator = true;
                break;
            }
        }
        if (!is_separator) {
            idx_interior.push_back(idx[i]);
        } else {
            is_separator = false; // reset for next index
        }
    }

    assert(idx_interior.size() + idx_separator.size() == static_cast<size_t>(size));

    SubdomainIdx subdomain_idx{idx_interior, idx_separator};
    return subdomain_idx;
}

struct SubdomainData
{
    std::vector<int> interior_sizes;
    std::vector<int> separator_sizes;
    std::vector<int> cluster_starts;
    uint* idx_h2;
    int size;
    int dim;
    int n_parts;
};

SubdomainData getSubdomainOrdering(Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int n_parts)
{
    int options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = 0;
    options[METIS_OPTION_MINCONN] = 1;

    // parition will have values between 0 and nParts-1 each value corresponding to a partition
    std::vector<int> partition = getPartitionKwayMetis(mat, n_parts, options);

    // get sizes of subdomains
    std::vector<int> subdomain_sizes(n_parts, 0);
    for (int i : partition) {
        subdomain_sizes[i]++;
    }
    std::vector<int> subdomain_pointers(n_parts, 0); // point to the beginning of each subdomain
    for (int i = 1; i < n_parts; i++) {
        subdomain_pointers[i] = subdomain_pointers[i - 1] + subdomain_sizes[i - 1];
    }
    // prn(subdomain_sizes);

    // sort according to partition (partition 0 first then partition 1, etc.)
    std::vector<int> idx(partition.size());
    for (size_t i = 0; i < partition.size(); i++) {
        idx[subdomain_pointers[partition[i]]] = i;
        subdomain_pointers[partition[i]]++;
    }

    // reset subdomain pointers to point to the beginning of each subdomain again
    subdomain_pointers.assign(n_parts, 0);
    for (int i = 1; i < n_parts; i++) {
        subdomain_pointers[i] = subdomain_pointers[i - 1] + subdomain_sizes[i - 1];
    }

    std::vector<int> idx_interior; idx_interior.reserve(partition.size());
    std::vector<int> idx_separator; idx_separator.reserve(partition.size());
    std::vector<int> interior_sizes(n_parts, 0);
    std::vector<int> separator_sizes(n_parts, 0);
    
    for (int i = 0; i < n_parts; i++) {
        auto [idx_interior_tmp, idx_separator_tmp] = getSubdomainIdx(mat, idx, subdomain_pointers[i], subdomain_sizes[i]);
        for (int j : idx_interior_tmp)
            idx_interior.push_back(j);
        for (int j : idx_separator_tmp)
            idx_separator.push_back(j);
        interior_sizes[i] = idx_interior_tmp.size();
        separator_sizes[i] = idx_separator_tmp.size();
    }

    std::vector<int> cluster_starts(2*n_parts + 1, 0);
    for (int i = 1; i <= n_parts; i++) {
        cluster_starts[i] = cluster_starts[i - 1] + interior_sizes[i - 1];
    }
    for (int i = 1; i <= n_parts; i++) {
        cluster_starts[n_parts + i] = cluster_starts[n_parts + i - 1] + separator_sizes[i - 1];
    }

    assert(idx_interior.size() + idx_separator.size() == partition.size());

    uint* idx_h2 = new uint[idx_interior.size() + idx_separator.size()];

    for (size_t i = 0; i < idx_interior.size(); i++)
        idx_h2[i] = idx_interior[i];

    for (size_t i = idx_interior.size(); i < idx_interior.size() + idx_separator.size(); i++)
        idx_h2[i] = idx_separator[i - idx_interior.size()];

    return SubdomainData{interior_sizes, separator_sizes, cluster_starts, idx_h2, static_cast<int>(idx_interior.size() + idx_separator.size()), 3, n_parts};
}

Eigen::SparseMatrix<int, Eigen::RowMajor> createAuxSparseMatrix(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, const std::vector<int>& cluster_sizes, 
    const std::vector<int>& leaf_clusters_per_son)
{
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(cluster_sizes.size());

    std::vector<int> block_ends(leaf_clusters_per_son.size());
    block_ends[0] = leaf_clusters_per_son[1];
    for (size_t i = 1; i < leaf_clusters_per_son.size(); i++) {
        block_ends[i] = block_ends[i-1] + leaf_clusters_per_son[i];
    }

    int row_offset = 0;
    int block_row_counter = 0;

    for (int i = 0; i < static_cast<int>(cluster_sizes.size()); i++) {
        int col_offset = 0;
        int block_col_counter = 0;
        if (i == block_ends[block_row_counter])
            block_row_counter++;
        for (int j = 0; j < static_cast<int>(cluster_sizes.size()); j++) {
            if (j == block_ends[block_col_counter])
                block_col_counter++;

            if (block_row_counter == block_col_counter)
                tripletList.emplace_back(i, j, 1);
            else {
                if (mat.block(row_offset, col_offset, cluster_sizes[i], cluster_sizes[j]).norm() != 0) {
                    tripletList.emplace_back(i, j, 1);
                }
            }
            col_offset += cluster_sizes[j];
        }
        row_offset += cluster_sizes[i];
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> aux_mat(cluster_sizes.size(), cluster_sizes.size());
    aux_mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return aux_mat;
}

void getLeafClusterSizesRecursive(pcluster c, std::vector<int>& sizes)
{
    if (c->sons == 0) {
        sizes.push_back(c->size);
    } else {
        for (uint i = 0; i < c->sons; i++) {
            getLeafClusterSizesRecursive(c->son[i], sizes);
        }
    }
}

std::vector<int> getLeafClusterSizes(pcluster c)
{
    std::vector<int> sizes;
    int depth = getmindepth_cluster(c);
    sizes.reserve(c->sons*std::pow(2, depth-1));
    getLeafClusterSizesRecursive(c, sizes);
    return sizes;
}

pcluster build_subdomain_cluster(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, 
    const SubdomainData& data, const ClusteringOptions& cluster_opt)
{
    pcluster c = new_cluster(data.size, data.idx_h2, 2*data.n_parts, data.dim);

    ClusteringDataVel data_vel = {
        mat,
        nullptr,
        std::vector<int>(),
        cluster_opt.max_leaf_size_vel,
        0,
        data.dim,
        cluster_opt.min_size_after_disection
    };

    for (int i = 0; i < data.n_parts; i++) {
        c->son[i] = build_dd_cluster_vel(data.interior_sizes[i], data.idx_h2 + data.cluster_starts[i], data_vel, {&partitionMetisWrapper}, {&computeSimpleSeparator});
    }
    int max_depth = 0;
    for (int i = 0; i < data.n_parts; i++) {
        int depth = getmindepth_cluster(c->son[i]);
        if (depth > max_depth)
            max_depth = depth;
    }
    for (int i = 0; i < data.n_parts; i++) {
        c->son[i+data.n_parts] = build_bisection_cluster_metis(data.separator_sizes[i], data.idx_h2 + data.cluster_starts[data.n_parts + i], data_vel, 
            data.separator_sizes[i], max_depth, 0);
    }

    std::vector<int> depths(data.n_parts);
    max_depth = 0;
    for (int i = 0; i < data.n_parts; i++) {
        depths[i] = getmindepth_cluster(c->son[i+data.n_parts]);
        if (depths[i] > max_depth)
            max_depth = depths[i];
    }

    for (int i = 0; i < data.n_parts; i++) {
        if (depths[i] < max_depth) {
            extend_cluster(c->son[i+data.n_parts], max_depth);
        }
    }

    update_cluster(c);

    int interior_size_total = data.cluster_starts[data.n_parts];
    int separator_size_total = data.size - interior_size_total;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(data.size);
    for (int i = 0; i < data.size; i++)
        P.indices()(data.idx_h2[i]) = i;
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_perm = P * mat * P.transpose();
    
    Eigen::SparseMatrix<int, Eigen::RowMajor> schur = mat_perm.block(interior_size_total, interior_size_total, separator_size_total, separator_size_total);

    std::vector<int> leaf_clusters_per_son(data.n_parts); // number of leaf clusters of the top level sons
    std::vector<int> leaf_cluster_sizes;
    leaf_cluster_sizes.reserve(data.n_parts * std::pow(2, max_depth-1));
    for (int i = data.n_parts; i < 2*data.n_parts; i++) {
        std::vector<int> temp = getLeafClusterSizes(c->son[i]);
        leaf_clusters_per_son[i-data.n_parts] = temp.size();
        for (int s : temp)
            leaf_cluster_sizes.push_back(s);
    }

    // Eigen::SparseMatrix<int, Eigen::RowMajor> aux_schur = createAuxSparseMatrix(schur, leaf_cluster_sizes, leaf_clusters_per_son);

    // SubdomainData data_schur = getSubdomainOrdering(aux_schur, 4);

    // Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P1(data_schur.size);
    // for (int i = 0; i < data_schur.size; i++)
    //     P1.indices()(data_schur.idx_h2[i]) = i;
    // Eigen::SparseMatrix<int, Eigen::RowMajor> aux_schur_perm = P1 * aux_schur * P1.transpose();

    // writeMatrix2File(aux_schur_perm.cast<double>(), "aux_schur");

    SubdomainData data_schur = getSubdomainOrdering(schur, 8);

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P1(data.size);
    for (int i = 0; i < interior_size_total; i++)
        P1.indices()(i) = i;
    for (int i = 0; i < data_schur.size; i++)
        P1.indices()(data_schur.idx_h2[i]+interior_size_total) = interior_size_total + i;
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_perm1 = P1 * mat_perm * P1.transpose();

    // writeMatrix2File(mat_perm1.cast<double>(), "aux_schur");

    return c;
}

pcluster getSubdomainClustering(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, 
    int n_parts, const mm::OseenDiscretizationBetter& dc, const ClusteringOptions& cluster_opt)
{
    // cut out velocity block from the matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = mat.block(0, 0, dc.N_ui + dc.N_uneu, dc.N_ui + dc.N_uneu).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 0, dc.N_ui + dc.N_uneu, dc.N_ui + dc.N_uneu).transpose().cast<int>());
    mat_vel.makeCompressed();

    SubdomainData data = getSubdomainOrdering(mat_vel, n_parts);
    data.dim = dc.d_u.dim;
    pcluster c = build_subdomain_cluster(mat_vel, data, cluster_opt);
    
    return c;
}

namespace {
// test functionality of findSurroundingGraph, bfsFindStart and build_bfs_interface_cluster
// void test_interface()
// {
//     std::vector<uint> idx = {0, 3, 4};
//     Eigen::MatrixXi mat_dense {
//         {1, 1, 1, 0, 0, 0},
//         {1, 1, 0, 1, 1, 0},
//         {1, 0, 1, 1, 1, 1},
//         {0, 1, 1, 1, 1, 0},
//         {0, 1, 1, 1, 1, 0},
//         {0, 0, 1, 0, 0, 1}
//     };
//     Eigen::SparseMatrix<int, Eigen::RowMajor> mat = mat_dense.sparseView();
//     std::vector<uint> neighbors;
//     for (int i = 0; i < 3; i++)
//         neighbors.push_back(idx[i]);
    
//     Eigen::SparseMatrix<int, Eigen::RowMajor> surroundingGraph = findSurroundingGraph(mat, neighbors);

//     std::vector<uint> neighbors_expect = {0, 3, 4, 1, 2};
//     assert(neighbors_expect.size() == neighbors.size());
//     assert(neighbors_expect == neighbors);
    

//     std::vector<int> start = bfsFindStart(surroundingGraph, 3);
//     std::vector<int> start_expect = {0, 2};
//     assert(start_expect == start);

//     build_bfs_interface_cluster(mat, 3, idx.data(), 1, 3, 1);
//     std::vector<uint> idx_expect = {0, 3, 4};
//     assert(idx_expect == idx);
// }

void testNearestNeighborMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, mm::OseenDiscretizationBetter& dc, int degree)
{
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = createNearestNeighborMatrixVelocity(dc, degree);
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_vel_exact = mat.block(0, 0, dc.N_ui + dc.N_uneu, dc.N_ui + dc.N_uneu);
    assert(mat_vel_exact.rows() == mat_vel.rows());
    assert(mat_vel_exact.cols() == mat_vel.cols());
    assert(mat_vel_exact.isCompressed());
    assert(mat_vel.isCompressed());

    for (int k=0; k<mat_vel_exact.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat_vel_exact,k); it; ++it)
        {
            if (mat_vel.coeff(it.row(), it.col()) == 0)
                std::cout << "Error: Nearest neighbor matrix has zero entry at (" << it.row() << ", " << it.col() << ")" << std::endl;
        }
}

} // anonymous namespace for test functions