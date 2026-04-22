#include "support_by_cluster.hpp"

static void sortByPartition(const std::vector<int>& partition, std::vector<int>& idx)
{
    // sort to have partition 0 first then partition 1
    int size0 = 0, size1 = 0;
    for (size_t i = 0; i < idx.size(); i++) {
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

static Eigen::SparseMatrix<int, Eigen::RowMajor> 
getSubGraph(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, const std::vector<int>& idx1, const std::vector<int>& idx2)
{
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(idx1.size()*(mat.nonZeros()/mat.rows()));
    std::vector<int> flag(mat.cols(), 0);
    for (size_t i = 0; i < idx2.size(); i++)
        flag[idx2[i]] = 1;

    for (size_t i = 0; i < idx1.size(); i++) {
        for (int j = mat.outerIndexPtr()[idx1[i]]; j < mat.outerIndexPtr()[idx1[i] + 1]; j++) {
            int neighbor = mat.innerIndexPtr()[j];
            if (flag[neighbor] == 1) {
                tripletList.push_back(Eigen::Triplet<int>(i, neighbor, 1));
            }
        }
    }

    Eigen::SparseMatrix<int, Eigen::ColMajor> temp(idx1.size(), mat.cols());
    temp.setFromTriplets(tripletList.begin(), tripletList.end());
    temp.makeCompressed();
    tripletList.clear();
    tripletList.reserve(idx1.size()*(temp.nonZeros()/temp.rows()));

    for (size_t i = 0; i < idx2.size(); i++) {
        for (int j = temp.outerIndexPtr()[idx2[i]]; j < temp.outerIndexPtr()[idx2[i] + 1]; j++) {
            int neighbor = temp.innerIndexPtr()[j];
            tripletList.push_back(Eigen::Triplet<int>(neighbor, i, 1));
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> sub_mat(idx1.size(), idx2.size());
    sub_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    sub_mat.makeCompressed();
    return sub_mat;
}

// helper function to determine for which indexes supports and weights have to updated
static std::vector<int> hasToBeUpdated(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, const std::vector<int>& considered, const std::vector<int>& allowed)
{
    std::vector<int> partition(mat.cols(), 0);
    for (int i : allowed)
        partition[i] = 1;

    std::vector<int> idx(mat.cols());
    for (size_t i = 0; i < idx.size(); i++)
        idx[i] = i;

    // not allowed idx are in partition 0 and will be sorted to the front
    sortByPartition(partition, idx);
    int offset = static_cast<int>(idx.size() - allowed.size());
    idx.resize(offset);
    // get matrix considered idx x not allowed idx
    Eigen::SparseMatrix<int, Eigen::RowMajor> subgraph = getSubGraph(mat, considered, idx);

    std::vector<int> to_be_updated;
    to_be_updated.reserve(considered.size()/4);

    for (size_t i = 0; i < considered.size(); i++) {
        for (int j = subgraph.outerIndexPtr()[i]; j < subgraph.outerIndexPtr()[i+1]; j++) {
            // connected to something outside of allowed -> has to updated
            to_be_updated.push_back(considered[i]);
            break;
        }
    }

    // for (int i : considered) {
    //     for (int j = mat.outerIndexPtr()[i]; j < mat.outerIndexPtr()[i+1]; j++) {
    //         int k = mat.innerIndexPtr()[j];
    //         if (std::find(allowed.begin(), allowed.end(), k) == allowed.end()) {
    //             // connected to something outside of allowed -> has to updated
    //             to_be_updated.push_back(i);
    //             break;
    //         }
    //     }
    // }

    return to_be_updated;
}

static bool isContained(const std::vector<int>& a, const std::vector<int>& b)
{
    for (auto k : a) {
        if (std::find(b.begin(), b.end(), k) == b.end()) {
            return false;
        }
    }
    return true;
}

static std::vector<int> intersection(const std::vector<int>& idx1, const std::vector<int>& idx2) {
    std::unordered_set<int> set2(idx2.begin(), idx2.end());
    std::vector<int> result;
    int size_estimate = std::min(idx1.size(), idx2.size());
    result.reserve(size_estimate);
    for (int i : idx1) {
        if (set2.count(i)) {
            result.push_back(i);
        }
    }
    return result;
}

void setSearchAmongForNodesLapConv(pcluster c, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, int N_ub, 
            std::vector<int>& interface_idx, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
{
    assert(c->type == 1);
    assert(c->sons > 1);

    if (c->sons == 3) {
        assert(c->son[2]->type==2);
        std::vector<int> tmp_for_nodes;
        tmp_for_nodes.reserve(c->son[2]->size);
        std::vector<int> tmp_search_among;
        tmp_search_among.reserve(c->size + interface_idx.size());
        for (uint j = 0; j < c->son[2]->size; j++) {
            tmp_for_nodes.push_back(c->son[2]->idx[j]);
            tmp_search_among.push_back(c->son[2]->idx[j]);
        }
        for (uint j = 0; j < c->son[0]->size + c->son[1]->size; j++) {
            tmp_search_among.push_back(c->idx[j]);
        }
        for (size_t j = 0; j < interface_idx.size(); j++) {
            tmp_search_among.push_back(interface_idx[j]);
        }
        std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);
        // this if is needed because if for_nodes is empty, all nodes in te domain will be used instead (see FindClosest.hpp operator())
        if (to_be_updated.size() > 0) {
            for (int& i : to_be_updated)
                i += N_ub;
            for (int& i : tmp_search_among)
                i += N_ub;
            // domain boundary points are always in search_among
            for (int j = 0; j < N_ub; j++) {
                tmp_search_among.push_back(j);
            }
            for_nodes.push_back(to_be_updated);
            search_among.push_back(tmp_search_among);
        }
    }

    std::vector<std::vector<int>> tmp_interface_idx(2);
    for (int k = 0; k < 2; k++) {
        tmp_interface_idx[k] = interface_idx;
        if (c->sons == 3) {
            tmp_interface_idx[k].reserve(tmp_interface_idx[k].size() + c->son[2]->size);
            for (uint i = 0; i < c->son[2]->size; i++) {
                tmp_interface_idx[k].push_back(c->son[2]->idx[i]);
            }
        }
    }

    for (int k = 0; k < 2; k++) {
        if (max_depth == 0 || c->son[k]->sons == 0 || c->son[k]->type == 3) { // has no sons or is bisection cluster
            std::vector<int> tmp_for_nodes;
            std::vector<int> tmp_search_among;
            tmp_for_nodes.reserve(c->son[k]->size);
            tmp_search_among.reserve(c->son[k]->size + tmp_interface_idx[k].size() + N_ub);
            for (uint j = 0; j < c->son[k]->size; j++) {
                tmp_for_nodes.push_back(c->son[k]->idx[j]);
                tmp_search_among.push_back(c->son[k]->idx[j]);
            }
            for (size_t j = 0; j < tmp_interface_idx[k].size(); j++) {
                tmp_search_among.push_back(tmp_interface_idx[k][j]);
            }
            std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);
            // this if is needed because if for_nodes is empty, all nodes in te domain will be used instead (see FindClosest.hpp operator())
            if (to_be_updated.size() == 0) {
                for (int& i : to_be_updated)
                    i += N_ub;
                for (int& i : tmp_search_among)
                    i += N_ub;
                // domain boundary points are always in search_among
                for (int j = 0; j < N_ub; j++) {
                    tmp_search_among.push_back(j);
                }
                for_nodes.push_back(to_be_updated);
                search_among.push_back(tmp_search_among);
            }
        } else {
            setSearchAmongForNodesLapConv(c->son[k], search_among, for_nodes, N_ub, tmp_interface_idx[k], mat, max_depth-1);
        }
    }
}

void setSearchAmongForNodesHyp(pcluster c, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, int N_ub, 
            std::vector<int>& interface_idx, int min_size)
{
    assert(c->type == 1);
    assert(c->sons > 1);

    std::vector<std::vector<int>> tmp_interface_idx(2);
    for (int k = 0; k < 2; k++) {
        tmp_interface_idx[k] = interface_idx;
        if (c->sons == 3) {
            for (uint i = 0; i < c->son[2]->size; i++) {
                tmp_interface_idx[k].push_back(c->son[2]->idx[i]);
            }
        }
    }

    for (int k = 0; k < 2; k++) {
        if (c->son[k]->sons == 0 || c->son[k]->type == 3) {
            std::vector<int> tmp;
            for (uint j = 0; j < c->son[k]->size; j++) {
                tmp.push_back(c->son[k]->idx[j]+N_ub);
            }
            for_nodes.push_back(tmp);
            for (size_t j = 0; j < tmp_interface_idx[k].size(); j++) {
                tmp.push_back(tmp_interface_idx[k][j]+N_ub);
            }
            // domain boundary points are always in search_among
            for (int j = 0; j < N_ub; j++) {
                tmp.push_back(j);
            }
            search_among.push_back(tmp);
        } else {
            setSearchAmongForNodesHyp(c->son[k], search_among, for_nodes, N_ub, tmp_interface_idx[k], min_size);
        }
    }

    if (c->sons == 3) {
        std::vector<int> tmp2;
        for (uint j = 0; j < c->son[2]->size; j++) {
            tmp2.push_back(c->son[2]->idx[j]+N_ub);
        }
        for_nodes.push_back(tmp2);
        for (uint j = 0; j < c->son[0]->size + c->son[1]->size; j++) {
            tmp2.push_back(c->idx[j]+N_ub);
        }
        for (size_t j = 0; j < interface_idx.size(); j++) {
            tmp2.push_back(interface_idx[j]+N_ub);
        }
        // domain boundary points are always in search_among
        for (int j = 0; j < N_ub; j++) {
            tmp2.push_back(j);
        }
        search_among.push_back(tmp2);
    }
}

void setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, 
        int N_u, int N_ub, uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
{
    assert(cv->sons > 1);
    assert(cp->sons == 2);

    if (cv->sons == 3) {
        assert(cv->son[2]->type == 2); // interface cluster
        std::vector<int> tmp_for_nodes;
        tmp_for_nodes.reserve(cv->son[2]->size);
        for (uint i = 0; i < cv->son[2]->size; i++) {
            tmp_for_nodes.push_back(cv->son[2]->idx[i]);
        }
        
        std::vector<int> tmp_search_among;
        tmp_search_among.reserve(cp->size);
        for (uint j = 0; j < cp->size; j++) {
            tmp_search_among.push_back(cp->idx[j]);
        }

        std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

        for (int& i : to_be_updated)
            i += N_ub;
        for (int& i : tmp_search_among)
            i += N_u;
        for_nodes.push_back(to_be_updated);
        search_among.push_back(tmp_search_among);
    }

    for (int k = 0; k < 2; k++) {
        // cp->son[k]->son[0,1] only accessed if cp->son[k]->sons == 2, this works because of how booleans are evaluated in C++ 
        if (max_depth == 0 || cv->son[k]->sons == 0 || cp->son[k]->sons == 0 
            || (cp->son[k]->sons == 2 && (cp->son[k]->son[0]->size < min_size || cp->son[k]->son[1]->size < min_size))) {
            std::vector<int> tmp_for_nodes;
            tmp_for_nodes.reserve(cv->son[k]->size);
            std::vector<int> tmp_search_among;
            tmp_search_among.reserve(cp->son[k]->size);
            for (uint i = 0; i < cv->son[k]->size; i++) {
                tmp_for_nodes.push_back(cv->son[k]->idx[i]);
            }
            for (uint j = 0; j < cp->son[k]->size; j++) {
                tmp_search_among.push_back(cp->son[k]->idx[j]);
            }
            std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

            for (int& i : to_be_updated)
                i += N_ub;
            for (int& i : tmp_search_among)
                i += N_u;
            for_nodes.push_back(to_be_updated);
            search_among.push_back(tmp_search_among);
        } else {
            setSearchAmongForNodesGrad(cv->son[k], cp->son[k], search_among, for_nodes, N_u, N_ub, min_size, mat, max_depth-1);
        }
    }
}

void setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<std::vector<int>>& search_among, std::vector<std::vector<int>>& for_nodes, 
            int N_u, int N_ub, std::vector<int> interface_idx, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
{
    assert(cv->sons > 1);
    assert(cp->sons == 2);
    
    if (cv->sons == 3) {
        for (uint i = 0; i < cv->son[2]->size; i++) {
            interface_idx.push_back(cv->son[2]->idx[i]);
        }
    }

    for (int k = 0; k < 2; k++) {
        if (max_depth == 0 || cv->son[k]->sons == 0 || cp->son[k]->sons == 0) {
            std::vector<int> tmp_for_nodes;
            tmp_for_nodes.reserve(cp->son[k]->size);
            std::vector<int> tmp_search_among = interface_idx;

            for (uint i = 0; i < cp->son[k]->size; i++) {
                tmp_for_nodes.push_back(cp->son[k]->idx[i]);
            }
            for (uint j = 0; j < cv->son[k]->size; j++) {
                tmp_search_among.push_back(cv->son[k]->idx[j]);
            }
            
            std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

            for (int& i : to_be_updated)
                i += N_u;
            for (int& i : tmp_search_among)
                i += N_ub;
            for (int j = 0; j < N_ub; j++)
                tmp_search_among.push_back(j);
            for_nodes.push_back(to_be_updated);
            search_among.push_back(tmp_search_among);
        } else {
            setSearchAmongForNodesDiv(cv->son[k], cp->son[k], search_among, for_nodes, N_u, N_ub, interface_idx, mat, max_depth-1);
        }
    }
}

Indices setSearchAmongForNodesOseen(mm::OseenDiscretizationBetter& dc, pcluster rootv, pcluster rootp, 
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, int max_depth)
{
    assert(rootp->size == static_cast<uint>(dc.d_p.size()));
    assert(rootv->size == static_cast<uint>(dc.d_u_int.size()));

    int N = dc.N_ui + dc.N_uneu;

    // cut out velocity block from the matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_vel = mat.block(0, 0, N, N).cast<int>()
                    + static_cast<Eigen::SparseMatrix<int, Eigen::RowMajor>>(mat.block(0, 0, N, N).transpose().cast<int>());
    mat_vel.makeCompressed();
    // cut out div and grad block from matrix
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div = mat.block(3*N, 0, dc.N_p, N).cast<int>();
    Eigen::SparseMatrix<int, Eigen::RowMajor> mat_grad = mat.block(0, 3*N, N, dc.N_p).cast<int>();
    mat_div.makeCompressed();
    mat_grad.makeCompressed();

    std::vector<int> interface_idx_div;
    std::vector<int> interface_idx_lap_conv;
    std::vector<int> interface_idx_hyp;

    std::vector<std::vector<int>> search_among_lap_conv;
    std::vector<std::vector<int>> search_among_hyp;
    std::vector<std::vector<int>> search_among_grad;
    std::vector<std::vector<int>> search_among_div;
    std::vector<std::vector<int>> for_nodes_lap_conv;
    std::vector<std::vector<int>> for_nodes_hyp;
    std::vector<std::vector<int>> for_nodes_grad;
    std::vector<std::vector<int>> for_nodes_div;

    setSearchAmongForNodesLapConv(rootv, search_among_lap_conv, for_nodes_lap_conv, dc.N_ub, interface_idx_lap_conv, mat_vel, max_depth);
    setSearchAmongForNodesGrad(rootv, rootp, search_among_grad, for_nodes_grad, dc.N_u, dc.N_ub, dc.n[dc.poly_grad], mat_grad, max_depth);
    setSearchAmongForNodesDiv(rootv, rootp, search_among_div, for_nodes_div, dc.N_u, dc.N_ub, interface_idx_div, mat_div, max_depth);

    int number_of_stencils_to_be_updated_lapconv = 0;
    for (size_t i = 0; i < for_nodes_lap_conv.size(); i++) {
        number_of_stencils_to_be_updated_lapconv += for_nodes_lap_conv[i].size();
    }
    // std::cout << "number of stencils to be updated (lap/conv): " << number_of_stencils_to_be_updated_lapconv << std::endl;
    int number_of_stencils_to_be_updated_grad = 0;
    for (size_t i = 0; i < for_nodes_grad.size(); i++) {
        number_of_stencils_to_be_updated_grad += for_nodes_grad[i].size();
    }
    // std::cout << "number of stencils to be updated (grad): " << number_of_stencils_to_be_updated_grad << std::endl;
    int number_of_stencils_to_be_updated_div = 0;
    for (size_t i = 0; i < for_nodes_div.size(); i++) {
        number_of_stencils_to_be_updated_div += for_nodes_div[i].size();
    }
    // std::cout << "number of stencils to be updated (div): " << number_of_stencils_to_be_updated_div << std::endl;
    // std::cout << "number of velocity dofs: " << N << std::endl;
    // std::cout << "number of pressure dofs: " << dc.N_p << std::endl;

    std::cout << "fraction of stencils that have to be updated: " 
    << static_cast<double>(number_of_stencils_to_be_updated_div + 
        number_of_stencils_to_be_updated_grad + number_of_stencils_to_be_updated_lapconv)
        /static_cast<double>(2*N + dc.N_p) << std::endl;

    Indices idx_to_be_updated;
    idx_to_be_updated.idxs_vel.reserve(number_of_stencils_to_be_updated_lapconv);
    idx_to_be_updated.idxs_grad.reserve(number_of_stencils_to_be_updated_grad);
    idx_to_be_updated.idxs_div.reserve(number_of_stencils_to_be_updated_div);

    for (size_t i = 0; i < for_nodes_lap_conv.size(); i++)
        for (int j : for_nodes_lap_conv[i])
            if (j < dc.N_ub + dc.N_ui) // do not include ghost nodes here, ghost nodes are handled via idx_neu_vel
                idx_to_be_updated.idxs_vel.push_back(j);

    for (size_t i = 0; i < for_nodes_grad.size(); i++)
        for (int j : for_nodes_grad[i]) // do not include ghost nodes here, ghost nodes are handled via idx_neu_grad
            if (j < dc.N_ub + dc.N_ui)
                idx_to_be_updated.idxs_grad.push_back(j);

    for (size_t i = 0; i < for_nodes_div.size(); i++)
        for (int j : for_nodes_div[i])
            idx_to_be_updated.idxs_div.push_back(j);

    idx_to_be_updated.idxs_neu_vel = intersection(idx_to_be_updated.idxs_vel, dc.idxs_neu);
    idx_to_be_updated.idxs_neu_grad = intersection(idx_to_be_updated.idxs_grad, dc.idxs_neu);

    if (dc.use_hyperviscosity) {
        dc.determineSupportsWithHyperv(search_among_lap_conv, search_among_grad, search_among_div, 
        for_nodes_lap_conv, for_nodes_grad, for_nodes_div);
    } else {
        dc.determineSupports(search_among_lap_conv, search_among_grad, search_among_div, 
        for_nodes_lap_conv, for_nodes_grad, for_nodes_div);
    }

    return idx_to_be_updated;
}