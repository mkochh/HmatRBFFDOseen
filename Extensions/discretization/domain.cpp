#include "domain.hpp"

#include <functional>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_set>
#include <Eigen/Sparse>

namespace mm {
    
    int binom(const int n, const int k) {
        double sum;

        if(n==0 || k==0) {
            sum = 1;
        } else {
            sum = binom(n-1,k-1)+binom(n-1,k);
        }

        if((n==1 && k==0) || (n==1 && k==1))
            sum = 1;
        if(k > n)
            sum = 0;

        return sum;
    }

    OseenDiscretizationBetter::OseenDiscretizationBetter(const PolyhedronShape<Vec3d>& shape, const RBFFDOptions& rbffd_opt)
        : d_u(shape)
        , d_p(shape)
        , d_grad(shape)
        , d_div(shape)
        , d_conv(shape)
        , d_u_int(shape)
        , d_hyp(shape)
        , poly_lap(rbffd_opt.poly_lap)
        , poly_conv(rbffd_opt.poly_conv)
        , poly_grad(rbffd_opt.poly_grad)
        , poly_div(rbffd_opt.poly_div)
        , poly_hyp(0)
        , n_hyp(0)
        , k_hyp(0)
        , nu(rbffd_opt.nu)
        , dx_u(rbffd_opt.dx_u)
        , seed(rbffd_opt.seed)
    {
        initialize(shape, rbffd_opt.neumann, rbffd_opt.step_size_scale, rbffd_opt.subset);
        setConvection(rbffd_opt.conv);
        setSolution(rbffd_opt.sol, rbffd_opt.domain_geometry);
        setHyperViscosity(rbffd_opt.use_hyperviscosity);
    }

        OseenDiscretizationBetter::OseenDiscretizationBetter(const PolyhedronShape<Vec3d>& shape, const RBFFDOptions& rbffd_opt,
            std::function<bool(const Vec3d&)> is_neumann, std::function<double(const Vec3d&)> dx_u_func)
        : d_u(shape)
        , d_p(shape)
        , d_grad(shape)
        , d_div(shape)
        , d_conv(shape)
        , d_u_int(shape)
        , d_hyp(shape)
        , poly_lap(rbffd_opt.poly_lap)
        , poly_conv(rbffd_opt.poly_conv)
        , poly_grad(rbffd_opt.poly_grad)
        , poly_div(rbffd_opt.poly_div)
        , poly_hyp(0)
        , n_hyp(0)
        , k_hyp(0)
        , nu(rbffd_opt.nu)
        , dx_u(rbffd_opt.dx_u)
        , seed(rbffd_opt.seed)
    {
        initialize(shape, is_neumann, rbffd_opt.step_size_scale, rbffd_opt.subset, dx_u_func);
        setConvection(rbffd_opt.conv);
        setSolution(rbffd_opt.sol, rbffd_opt.domain_geometry);
        setHyperViscosity(rbffd_opt.use_hyperviscosity);
    }

    void OseenDiscretizationBetter::determineSupports() {
        FindClosest supp_lap(n[poly_lap]);
        supp_lap.forNodes(idxs_ui); // don't need stencils at boundary
        d_u.findSupport(supp_lap); // find support for lap within d_u
        FindClosest supp_conv(n[poly_conv]);
        supp_conv.forNodes(idxs_ui); // don't need stencils at boundary
        d_conv.findSupport(supp_conv); // find support for conv within d_conv(=d_u)
        FindClosest supp_grad(n[poly_grad]); // find support for grad over nodes in d_p at points in d_u
        supp_grad.forNodes(idxs_ui).searchAmong(idxs_p).forceSelf(false);
        d_grad.findSupport(supp_grad);
        FindClosest supp_divu(n[poly_div]); // find support for div u over nodes in d_u at points in d_p
        supp_divu.forNodes(idxs_p).searchAmong(idxs_u).forceSelf(false);
        d_div.findSupport(supp_divu);
    }

    void Dijkstra::reset_visited() {
        for (int i : visited_idx)
            visited[i] = false;
        visited_idx.clear();
    }

    void Dijkstra::reset_dist() {
        for (int i : dist_idx)
            dist[i] = std::numeric_limits<double>::infinity();
        dist_idx.clear();
    }

    Range<int> Dijkstra::n_nearest(int start, int n) {
        assert(0 <= start && start < graph.rows());
        using Pair = std::pair<double, int>; // (distance, node)
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;

        dist[start] = 0.0;
        dist_idx.reserve(n);
        visited_idx.reserve(n);
        dist_idx.push_back(start);
        pq.emplace(0.0, start);

        Range<int> result; result.reserve(n);
        while (!pq.empty() && static_cast<int>(result.size()) < n) {
            auto [d, u] = pq.top();
            pq.pop();
            if (visited[u]) continue;
            visited[u] = true;
            visited_idx.push_back(u);
            result.push_back(u);

            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, u); it; ++it) {
                int v = it.col();
                double weight = it.value();
                if (!visited[v] && dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    dist_idx.push_back(v);
                    pq.emplace(dist[v], v);
                }
            }
        }

        reset_dist();
        reset_visited();

        assert(static_cast<int>(result.size()) >= n);

        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    Range<int> Dijkstra::n_nearest_constrained(int start, int n, std::vector<int> flag) {
        assert(0 <= start && start < graph.rows());
        using Pair = std::pair<double, int>; // (distance, node)
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;

        dist[start] = 0.0;
        dist_idx.reserve(n);
        visited_idx.reserve(n);
        dist_idx.push_back(start);
        pq.emplace(0.0, start);

        explored_edges.push_back(0);

        Range<int> result; result.reserve(n);
        while (!pq.empty() && static_cast<int>(result.size()) < n) {
            auto [d, u] = pq.top();
            pq.pop();
            if (visited[u]) continue;
            visited[u] = true;
            visited_idx.push_back(u);
            if (flag[u] == 1)
                result.push_back(u);

            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, u); it; ++it) {
                int v = it.col();
                explored_edges.back()++;
                if (flag[v] == 1) {
                    double weight = it.value();
                    if (!visited[v] && dist[v] > dist[u] + weight) {
                        dist[v] = dist[u] + weight;
                        dist_idx.push_back(v);
                        pq.emplace(dist[v], v);
                    }
                }
            }
        }

        reset_dist();
        reset_visited();

        assert(static_cast<int>(result.size()) >= n || !(std::cerr << result.size() << "<" << n));

        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    void OseenDiscretizationBetter::determineSupportsGraph()
    {
        std::vector<Eigen::Triplet<double>> tripletList;
        int degree = 10;
        tripletList.reserve(d_u.size() * degree);
        Range<int> for_which_vv = d_u.all();
        KDTree<Vec3d> tree_v(d_u.positions());
        for (int i : for_which_vv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_u.pos(i), degree);
            // double r = std::sqrt(result.second[1]); if (std::sqrt(result.second[j]) < 2*r)
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j], std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_vv(d_u.size(), d_u.size());
        graph_vv.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_vv.makeCompressed();

        // for each node in idxs_ui determine support for lap/conv from graph by finding nearest neighbors using dijkstra
        Range<int> first_n_conv = Range<int>::seq(n[poly_conv]);
        assert(n[poly_conv] <= n[poly_lap]);

        Dijkstra dijkstra_vv(graph_vv);
        for (int i : idxs_ui) {
            d_u.support(i) = dijkstra_vv.n_nearest(i, n[poly_lap]);
            d_conv.support(i) = d_u.support(i)[first_n_conv];
        }

        Range<int> for_which_pv = d_p.all();
        for (int i : for_which_pv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_p.pos(i), degree);
            // double r = std::sqrt(result.second[2]); if (std::sqrt(result.second[j]) < 2*r)
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j], std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_pv(d_p.size() + d_u.size(), d_u.size());
        graph_pv.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_pv.makeCompressed();

        // for each node in idxs_p determine support for div from graph by finding nearest neighbors using dijkstra
        Dijkstra dijkstra_pv(graph_pv);
        for (int i : idxs_p)
            d_div.support(i) = dijkstra_pv.n_nearest(i, n[poly_div]+1)[Range<int>::seq(1, n[poly_div]+1)]; // skip self

        tripletList.clear();
        tripletList.reserve((d_u.size() + d_p.size()) * degree);

        KDTree<Vec3d> tree_p(d_p.positions());
        for (int i : d_p.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_p.pos(i), degree);
            // double r = std::sqrt(result.second[1]); if (std::sqrt(result.second[j]) < 2*r)
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }
        for (int i : d_u.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_u.pos(i), degree);
            // double r = std::sqrt(result.second[2]); if (std::sqrt(result.second[j]) < 2*r)
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_vp(d_p.size() + d_u.size(), d_p.size() + d_u.size());
        graph_vp.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_vp.makeCompressed();

        Dijkstra dijkstra_vp(graph_vp);
        for (int i : idxs_ui)
            d_grad.support(i) = dijkstra_vp.n_nearest(i, n[poly_grad]+1)[Range<int>::seq(1, n[poly_grad] + 1)]; // skip self
    }

    std::vector<int> intersection(const std::vector<int>& idx1, const std::unordered_set<int>& set2, int max) {
        std::vector<int> result;
        result.reserve(max);
        int counter = 0;
        for (int i : idx1) {
            if (set2.count(i)) {
                result.push_back(i);
                counter++;
                if (counter == max)
                    break;
            }
        }
        return result;
    }

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

        return to_be_updated;
    }

    static std::vector<int> hasToBeUpdated(const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, const std::vector<int>& considered, const std::vector<int>& allowed,
        int threshold)
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
            int counter = 0;
            for (int j = subgraph.outerIndexPtr()[i]; j < subgraph.outerIndexPtr()[i+1]; j++) {
                // connected to something outside of allowed -> has to updated
                counter++;
            }
            if (counter < threshold && counter > 0) {
                to_be_updated.push_back(considered[i]);
            }
            // if (counter > 0)
            //     prn(counter);
        }

        return to_be_updated;
    }

    void OseenDiscretizationBetter::setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
        std::vector<int> flag, Dijkstra& dijkstra, std::vector<int> interface_idx, 
        const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
    {
        assert(cv->sons > 1);
        assert(cp->sons == 2);

        if (cv->sons == 3) {
            for (uint i = 0; i < cv->son[2]->size; i++) {
                flag[cv->son[2]->idx[i] + N_ub] = 1;
                interface_idx.push_back(cv->son[2]->idx[i]);
            }
        }

        for (int k = 0; k < 2; k++) {
            if (max_depth == 0 || cv->son[k]->sons == 0 || cp->son[k]->sons == 0) {
                std::vector<int> tmp_for_nodes;
                tmp_for_nodes.reserve(cp->son[k]->size);
                std::vector<int> tmp_search_among = interface_idx;

                std::vector<int> tmp_flag = flag;

                for (uint i = 0; i < cp->son[k]->size; i++) {
                    tmp_for_nodes.push_back(cp->son[k]->idx[i]);
                }
                for (uint j = 0; j < cv->son[k]->size; j++) {
                    tmp_search_among.push_back(cv->son[k]->idx[j]);
                    tmp_flag[cv->son[k]->idx[j] + N_ub] = 1; 
                }
                
                std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

                for (int& i : to_be_updated)
                    i += N_u;
                
                for (int i : to_be_updated) {
                    d_div.support(i) = dijkstra.n_nearest_constrained(i, n[poly_div], tmp_flag);
                    to_be_updated_all.push_back(i);
                }
            } else {
                setSearchAmongForNodesDiv(cv->son[k], cp->son[k], to_be_updated_all, flag, dijkstra, interface_idx, mat, max_depth-1);
            }
        }
    }

    // flag should be all ones for pressure nodes [d_u.size(), d_u.size()+d_p.size) and 0 for velocity nodes [0,d_u.size())
    void OseenDiscretizationBetter::setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
        std::vector<int>& flag, Dijkstra& dijkstra,
        uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
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
                assert(flag[cp->idx[j] + d_u.size()] == 1);
            }

            std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

            for (int& i : to_be_updated)
                i += N_ub;
            for (int i : to_be_updated) {
                d_grad.support(i).clear();
                d_grad.support(i) = dijkstra.n_nearest_constrained(i, n[poly_grad], flag);
                to_be_updated_all.push_back(i);

                if (d_grad.support(i).size() < n[poly_grad]) {
                    std::cout << d_grad.support(i).size() << " < " << n[poly_grad] << std::endl;

                    // std::ofstream file("error_log.m", std::ofstream::app);

                    // file << "pos_grad = " << d_grad.positions() << ";" << std::endl;
                    // file << "start = " << d_grad.pos(i) << ";" << std::endl;
                    // file << "idx_supp = " << d_grad.support(i) << ";" << std::endl;
                    // file << "idx_flag = " << tmp_search_among << ";" << std::endl;
                    // file << "flag = " << flag << ";" << std::endl;

                    // file.close();

                    // exit(1);
                }
            }
        }

        std::array<std::vector<int>, 2> tmp_flags;

        for (int k = 0; k < 2; k++) {
            tmp_flags[k] = flag;
            for (uint i = 0; i < cp->son[(k+1)%2]->size; i++)
                tmp_flags[k][cp->son[(k+1)%2]->idx[i] + d_u.size()] = 0;
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
                    assert(flag[cp->son[k]->idx[j] + d_u.size()] == 1);
                }
                std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

                for (int& i : to_be_updated)
                    i += N_ub;
                for (int i : to_be_updated) {
                    d_grad.support(i).clear();
                    d_grad.support(i) = dijkstra.n_nearest_constrained(i, n[poly_grad], tmp_flags[k]);
                    to_be_updated_all.push_back(i);

                    if (d_grad.support(i).size() < n[poly_grad]) {
                        std::cout << d_grad.support(i).size() << " < " << n[poly_grad] << std::endl;
                        // std::ofstream file("error_log.m", std::ofstream::app);

                        // file << "pos_grad = " << d_grad.positions() << ";" << std::endl;
                        // file << "start = " << d_grad.pos(i) << ";" << std::endl;
                        // file << "idx_supp = " << d_grad.support(i) << ";" << std::endl;
                        // file << "idx_flag = " << tmp_search_among << ";" << std::endl;
                        // file << "flag = " << flag << ";" << std::endl;

                        // file.close();

                        // exit(1);
                    }
                }
            } else {
                setSearchAmongForNodesGrad(cv->son[k], cp->son[k], to_be_updated_all, tmp_flags[k], dijkstra, min_size, mat, max_depth-1);
            }
        }
    }

    Indices OseenDiscretizationBetter::determineSupportsGraph(pcluster rootv, pcluster rootp, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
        int max_depth)
    {
        assert(rootp->size == static_cast<uint>(d_p.size()));
        assert(rootv->size == static_cast<uint>(d_u_int.size()));

        int N = N_ui + N_uneu;

        // cut out div and grad block from matrix
        Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div = mat.block(3*N, 0, N_p, N).cast<int>();
        Eigen::SparseMatrix<int, Eigen::RowMajor> mat_grad = mat.block(0, 3*N, N, N_p).cast<int>();
        mat_div.makeCompressed();
        mat_grad.makeCompressed();

        std::vector<Eigen::Triplet<double>> tripletList;
        int degree_vp = n[poly_grad];
        tripletList.reserve(d_u.size() * degree_vp);
        Range<int> for_which_vv = d_u.all();
        KDTree<Vec3d> tree_v(d_u.positions());
        for (int i : for_which_vv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_u.pos(i), degree_vp);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j], std::sqrt(result.second[j])));
        }

        Range<int> for_which_pv = d_p.all();
        for (int i : for_which_pv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_p.pos(i), degree_vp);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j], std::sqrt(result.second[j])));
        }

        // KDTree<Vec3d> tree_v(d_u.positions());
        // for (int i : for_which_vv) {
        //     std::pair<Range<int>, Range<double>> result = tree_v.query(d_u.pos(i), degree_vp);
        //     double r = std::sqrt(result.second[1]);
        //     for (int j = 0; j < static_cast<int>(result.first.size()); j++)
        //         if (std::sqrt(result.second[j]) < 2*r)
        //             tripletList.push_back(Eigen::Triplet<double>(i, result.first[j], std::sqrt(result.second[j])));
        // }

        // Range<int> for_which_pv = d_p.all();
        // for (int i : for_which_pv) {
        //     std::pair<Range<int>, Range<double>> result = tree_v.query(d_p.pos(i), degree_vp);
        //     double r = std::sqrt(result.second[2]);
        //     for (int j = 0; j < static_cast<int>(result.first.size()); j++)
        //         if (std::sqrt(result.second[j]) < 2*r)
        //             tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j], std::sqrt(result.second[j])));
        // }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_pv(d_p.size() + d_u.size(), d_u.size());
        graph_pv.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_pv.makeCompressed();

        // for for_nodes_grad determine support for div from graph by finding nearest neighbors using dijkstra
        Dijkstra dijkstra_pv(graph_pv);
        std::vector<int> flag_pv(d_u.size() + d_p.size(), 0);
        for (int i = 0; i < N_ub; i++) // dirichlet boundary nodes are always allowed
            flag_pv[i] = 1;
        std::vector<int> to_be_updated_div; to_be_updated_div.reserve(0.25*N_p);
        std::vector<int> interface_idx;

        setSearchAmongForNodesDiv(rootv, rootp, to_be_updated_div, flag_pv, dijkstra_pv, interface_idx, mat_div, max_depth);        

        int degree_pv = n[poly_div];
        tripletList.clear();
        tripletList.reserve((d_u.size() + d_p.size()) * degree_pv);

        KDTree<Vec3d> tree_p(d_p.positions());
        for (int i : d_p.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_p.pos(i), degree_pv);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }
        for (int i : d_u.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_u.pos(i), degree_pv);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }

        // KDTree<Vec3d> tree_p(d_p.positions());
        // for (int i : d_p.all()) {
        //     std::pair<Range<int>, Range<double>> result = tree_p.query(d_p.pos(i), degree_pv);
        //     double r = std::sqrt(result.second[1]);
        //     for (int j = 0; j < static_cast<int>(result.first.size()); j++)
        //         if (std::sqrt(result.second[j]) < 2*r)
        //             tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        // }

        // for (int i : d_u.all()) {
        //     std::pair<Range<int>, Range<double>> result = tree_p.query(d_u.pos(i), degree_pv);
        //     double r = std::sqrt(result.second[2]);
        //     for (int j = 0; j < static_cast<int>(result.first.size()); j++)
        //         if (std::sqrt(result.second[j]) < 2*r)
        //             tripletList.push_back(Eigen::Triplet<double>(i, result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        // }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_vp(d_p.size() + d_u.size(), d_p.size() + d_u.size());
        graph_vp.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_vp.makeCompressed();

        Dijkstra dijkstra_vp(graph_vp);

        std::vector<int> flag_vp(d_u.size() + d_p.size(), 0);
        for (size_t i = d_u.size(); i < flag_vp.size(); i++)
            flag_vp[i] = 1;
        std::vector<int> to_be_updated_grad; to_be_updated_grad.reserve(0.25*N_u);

        setSearchAmongForNodesGrad(rootv, rootp, to_be_updated_grad, flag_vp, dijkstra_vp, 2*n[poly_grad], mat_grad, max_depth);

        Indices to_be_updated;

        to_be_updated.idxs_grad.reserve(to_be_updated_grad.size());
        to_be_updated.idxs_neu_grad.reserve(N_uneu);
        for (int i : to_be_updated_grad)
            if(i < N_ui + N_ub) // if not a ghost node
                to_be_updated.idxs_grad.push_back(i);
            else
                to_be_updated.idxs_neu_grad.push_back(i);

        to_be_updated.idxs_div = to_be_updated_div;

        std::cout << "Fraction of stencils to be updated (div): " << static_cast<double>(to_be_updated_div.size())/static_cast<double>(d_p.size()) << std::endl;
        std::cout << "Fraction of stencils to be updated (grad): " << static_cast<double>(to_be_updated_grad.size())/static_cast<double>(N_ui + N_uneu) << std::endl;

        std::cout << "Average explored edges div: " << std::reduce(dijkstra_pv.explored_edges.begin(), dijkstra_pv.explored_edges.end())/dijkstra_pv.explored_edges.size() << std::endl;
        std::cout << "Average explored edges grad: " << std::reduce(dijkstra_vp.explored_edges.begin(), dijkstra_vp.explored_edges.end())/dijkstra_vp.explored_edges.size() << std::endl;

        std::cout << "Max explored edges div: " << *std::max_element(dijkstra_pv.explored_edges.begin(), dijkstra_pv.explored_edges.end()) << std::endl;
        std::cout << "Max explored edges grad: " << *std::max_element(dijkstra_vp.explored_edges.begin(), dijkstra_vp.explored_edges.end()) << std::endl;

        return to_be_updated;
    }

    void OseenDiscretizationBetter::setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
        std::vector<int> flag, const KDTree<Vec3d>& tree, std::vector<int> interface_idx, 
        const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
    {
        assert(cv->sons > 1);
        assert(cp->sons == 2);

        if (cv->sons == 3) {
            for (uint i = 0; i < cv->son[2]->size; i++) {
                flag[cv->son[2]->idx[i] + N_ub] = 1;
                interface_idx.push_back(cv->son[2]->idx[i]);
            }
        }

        for (int k = 0; k < 2; k++) {
            if (max_depth == 0 || cv->son[k]->sons == 0 || cp->son[k]->sons == 0) {
                std::vector<int> tmp_for_nodes;
                tmp_for_nodes.reserve(cp->son[k]->size);
                std::vector<int> tmp_search_among = interface_idx;

                std::vector<int> tmp_flag = flag;

                for (uint i = 0; i < cp->son[k]->size; i++) {
                    tmp_for_nodes.push_back(cp->son[k]->idx[i]);
                }
                for (uint j = 0; j < cv->son[k]->size; j++) {
                    tmp_search_among.push_back(cv->son[k]->idx[j]);
                    tmp_flag[cv->son[k]->idx[j] + N_ub] = 1; 
                }
                
                std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

                for (int& i : to_be_updated)
                    i += N_u;
                
                for (int i : to_be_updated) {
                    int factor = 3;
                    do {
                        d_div.support(i).clear();
                        Range<int> nearest = tree.query(d_div.pos(i), std::min(factor*n[poly_div], tree.size())).first;
                        // d_div.support(i).reserve(n[poly_div]); .clear() does not change capacity
                        for (int j : nearest) {
                            if (tmp_flag[j] == 1) 
                                d_div.support(i).push_back(j);
                            if (d_div.support(i).size() == n[poly_div])
                                break;
                        }
                        factor *= 2;
                    } while (d_div.support(i).size() < 0.8*n[poly_div]);

                    to_be_updated_all.push_back(i);
                }
            } else {
                setSearchAmongForNodesDiv(cv->son[k], cp->son[k], to_be_updated_all, flag, tree, interface_idx, mat, max_depth-1);
            }
        }
    }

    // flag should be all ones for pressure nodes [d_u.size(), d_u.size()+d_p.size) and 0 for velocity nodes [0,d_u.size())
    void OseenDiscretizationBetter::setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
        std::vector<int>& flag, const KDTree<Vec3d>& tree,
        uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth)
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
                assert(flag[cp->idx[j] + d_u.size()] == 1);
            }

            std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

            for (int& i : to_be_updated)
                i += N_ub;
            for (int i : to_be_updated) {
                int factor = 3;
                do {
                    d_grad.support(i).clear();
                    Range<int> nearest = tree.query(d_grad.pos(i), std::min(factor*n[poly_grad], tree.size())).first;
                    // d_grad.support(i).reserve(n[poly_grad]); .clear() does not change capacity
                    for (int j : nearest) {
                        if (flag[j + d_u.size()] == 1) 
                            d_grad.support(i).push_back(j + d_u.size());
                        if (d_grad.support(i).size() == n[poly_grad])
                            break;
                    }
                    factor *= 2;
                } while (d_grad.support(i).size() < 0.8*n[poly_grad]);

                to_be_updated_all.push_back(i);
            }
        }

        std::array<std::vector<int>, 2> tmp_flags;

        for (int k = 0; k < 2; k++) {
            tmp_flags[k] = flag;
            for (uint i = 0; i < cp->son[(k+1)%2]->size; i++)
                tmp_flags[k][cp->son[(k+1)%2]->idx[i] + d_u.size()] = 0;
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
                    assert(tmp_flags[k][cp->son[k]->idx[j] + d_u.size()] == 1);
                }
                std::vector<int> to_be_updated = hasToBeUpdated(mat, tmp_for_nodes, tmp_search_among);

                for (int& i : to_be_updated)
                    i += N_ub;
                for (int i : to_be_updated) {
                    int factor = 3;
                    do {
                        d_grad.support(i).clear();
                        Range<int> nearest = tree.query(d_grad.pos(i), std::min(factor*n[poly_grad], tree.size())).first;
                        // d_grad.support(i).reserve(n[poly_grad]); .clear() does not change capacity
                        for (int j : nearest) {
                            if (tmp_flags[k][j + d_u.size()] == 1) 
                                d_grad.support(i).push_back(j + d_u.size());
                            if (d_grad.support(i).size() == n[poly_grad])
                                break;
                        }
                        factor *= 2;
                    } while (d_grad.support(i).size() < 0.8*n[poly_grad]);

                    to_be_updated_all.push_back(i);
                }
            } else {
                setSearchAmongForNodesGrad(cv->son[k], cp->son[k], to_be_updated_all, tmp_flags[k], tree, min_size, mat, max_depth-1);
            }
        }
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

    Indices OseenDiscretizationBetter::determineSupportsGreedy(pcluster rootv, pcluster rootp, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
        int max_depth)
    {
        assert(rootp->size == static_cast<uint>(d_p.size()));
        assert(rootv->size == static_cast<uint>(d_u_int.size()));

        int N = N_ui + N_uneu;

        // cut out div and grad block from matrix
        Eigen::SparseMatrix<int, Eigen::RowMajor> mat_div = mat.block(3*N, 0, N_p, N).cast<int>();
        Eigen::SparseMatrix<int, Eigen::RowMajor> mat_grad = mat.block(0, 3*N, N, N_p).cast<int>();
        mat_div.makeCompressed();
        mat_grad.makeCompressed();
        
        KDTree<Vec3d> tree_div(d_u.positions());
        KDTree<Vec3d> tree_grad(d_p.positions());

        // for for_nodes_grad determine support for div from graph by finding nearest neighbors using dijkstra
        std::vector<int> flag_div(d_u.size() + d_p.size(), 0);
        for (int i = 0; i < N_ub; i++) // dirichlet boundary nodes are always allowed
            flag_div[i] = 1;
        std::vector<int> to_be_updated_div; to_be_updated_div.reserve(0.25*N_p);
        std::vector<int> interface_idx;

        setSearchAmongForNodesDiv(rootv, rootp, to_be_updated_div, flag_div, tree_div, interface_idx, mat_div, max_depth);

        std::vector<int> flag_grad(d_u.size() + d_p.size(), 0);
        for (size_t i = d_u.size(); i < flag_grad.size(); i++)
            flag_grad[i] = 1;
        std::vector<int> to_be_updated_grad; to_be_updated_grad.reserve(0.25*N_u);

        setSearchAmongForNodesGrad(rootv, rootp, to_be_updated_grad, flag_grad, tree_grad, 2*n[poly_grad], mat_grad, max_depth);

        Indices to_be_updated;

        to_be_updated.idxs_grad.reserve(to_be_updated_grad.size());
        to_be_updated.idxs_neu_grad.reserve(N_uneu);
        for (int i : to_be_updated_grad) {
            if (i < N_ui + N_ub) // interior node (includes neumann nodes)
                to_be_updated.idxs_grad.push_back(i);
            if (i >= N_ui + N_ub) // ghost node
                to_be_updated.idxs_neu_grad.push_back(i-N_ui); // push back neumann node which corresponds to ghost node i
        }

        to_be_updated.idxs_neu_grad = intersection(to_be_updated.idxs_grad, idxs_neu);

        to_be_updated.idxs_div = to_be_updated_div;

        std::cout << "Fraction of stencils to be updated (div): " << static_cast<double>(to_be_updated_div.size())/static_cast<double>(d_p.size()) << std::endl;
        std::cout << "Fraction of stencils to be updated (grad): " << static_cast<double>(to_be_updated_grad.size())/static_cast<double>(N_ui + N_uneu) << std::endl;

        return to_be_updated;
    }

    void OseenDiscretizationBetter::determineSupportsGraph(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div)
    {
        std::vector<Eigen::Triplet<double>> tripletList;
        int degree = 60;
        tripletList.reserve(d_u.size() * degree);
        Range<int> for_which_vv = d_u.all();
        KDTree<Vec3d> tree_v(d_u.positions());
        for (int i : for_which_vv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_u.pos(i), degree);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j], std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_vv(d_u.size(), d_u.size());
        graph_vv.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_vv.makeCompressed();

        // for each node in idxs_ui determine support for lap/conv from graph by finding nearest neighbors using dijkstra
        Range<int> first_n_conv = Range<int>::seq(n[poly_conv]);
        assert(n[poly_conv] <= n[poly_lap]);

        Dijkstra dijkstra_vv(graph_vv);
        std::vector<int> flag_vv(d_u.size(), 0);
        for (size_t i = 0; i < for_nodes_lap_conv.size(); i++) {
            if (for_nodes_lap_conv[i].size() == 0)
                continue;
            for (int j : search_among_lap_conv[i])
                flag_vv[j] = 1;
            for (int j : for_nodes_lap_conv[i]) {
                d_u.support(j) = dijkstra_vv.n_nearest_constrained(j, n[poly_lap], flag_vv);
                d_conv.support(j) = d_u.support(j)[first_n_conv];
            }
            for (int j : search_among_lap_conv[i])
                flag_vv[j] = 0;
        }

        Range<int> for_which_pv = d_p.all();
        for (int i : for_which_pv) {
            std::pair<Range<int>, Range<double>> result = tree_v.query(d_p.pos(i), degree);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j], std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_pv(d_p.size() + d_u.size(), d_u.size());
        graph_pv.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_pv.makeCompressed();

        // for for_nodes_grad determine support for div from graph by finding nearest neighbors using dijkstra
        Dijkstra dijkstra_pv(graph_pv);
        std::vector<int> flag_pv(d_u.size() + d_p.size(), 0);
        for (size_t i = 0; i < for_nodes_div.size(); i++) {
            if (for_nodes_div[i].size() == 0)
                continue;
            for (int j : search_among_div[i])
                flag_pv[j] = 1;
            for (int j : for_nodes_div[i]) {
                d_div.support(j) = dijkstra_pv.n_nearest_constrained(j, n[poly_div], flag_pv); // skip self
                for (int k : d_div.support(j))
                    assert(0 <= k && k < d_u.size());
            }
            for (int j : search_among_div[i])
                flag_pv[j] = 0;
        }

        tripletList.clear();
        tripletList.reserve((d_u.size() + d_p.size()) * degree);

        KDTree<Vec3d> tree_p(d_p.positions());
        for (int i : d_p.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_p.pos(i), degree);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i + d_u.size(), result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }
        for (int i : d_u.all()) {
            std::pair<Range<int>, Range<double>> result = tree_p.query(d_u.pos(i), degree);
            for (int j = 0; j < static_cast<int>(result.first.size()); j++)
                tripletList.push_back(Eigen::Triplet<double>(i, result.first[j] + d_u.size(), std::sqrt(result.second[j])));
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> graph_vp(d_p.size() + d_u.size(), d_p.size() + d_u.size());
        graph_vp.setFromTriplets(tripletList.begin(), tripletList.end());
        graph_vp.makeCompressed();

        Dijkstra dijkstra_vp(graph_vp);

        std::vector<int> flag_vp(d_u.size() + d_p.size(), 0);
        for (size_t i = 0; i < for_nodes_grad.size(); i++) {
            if (for_nodes_grad[i].size() == 0)
                continue;
            for (int j : search_among_grad[i])
                flag_vp[j] = 1;
            for (int j : for_nodes_grad[i]) {
                d_grad.support(j) = dijkstra_vp.n_nearest_constrained(j, n[poly_grad], flag_vp); // skip self
                for (int k : d_grad.support(j))
                    assert(d_u.size() <= k && k < d_u.size() + d_p.size());
            }
            for (int j : search_among_grad[i])
                flag_vp[j] = 0;
        }
    }

    void OseenDiscretizationBetter::determineSupports(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div)
    {
        // boundary nodes need to be in search_among for lap/conv/div
        assert(search_among_lap_conv.size() == for_nodes_lap_conv.size());
        assert(search_among_grad.size() == for_nodes_grad.size());
        assert(search_among_div.size() == for_nodes_div.size());

        FindClosest supp_lap(n[poly_lap]);
        FindClosest supp_conv(n[poly_conv]);
        mm::Range<int> first_n_conv = mm::Range<int>::seq(n[poly_conv]);
        assert(n[poly_conv] <= n[poly_lap]);
        for (size_t i = 0; i < for_nodes_lap_conv.size(); i++) {
            if (for_nodes_lap_conv[i].size() == 0)
                continue;
            supp_lap.forNodes(for_nodes_lap_conv[i]).searchAmong(search_among_lap_conv[i]);
            d_u.findSupport(supp_lap);
            for (int j : for_nodes_lap_conv[i]) {
                d_conv.support(j) = d_u.support(j)[first_n_conv];
            }
        }

        FindClosest supp_grad(n[poly_grad]); // find support for grad over nodes in d_p at points in d_u
        for (size_t i = 0; i < for_nodes_grad.size(); i++) {
            if (for_nodes_grad[i].size() == 0)
                continue;
            supp_grad.forNodes(for_nodes_grad[i]).searchAmong(search_among_grad[i]);
            d_grad.findSupport(supp_grad);
        }

        FindClosest supp_divu(n[poly_div]); // find support for div u over nodes in d_u at points in d_p
        for (size_t i = 0; i < for_nodes_div.size(); i++) {
            if (for_nodes_div[i].size() == 0)
                continue;
            supp_divu.forNodes(for_nodes_div[i]).searchAmong(search_among_div[i]);
            d_div.findSupport(supp_divu);
        }
    }

    void OseenDiscretizationBetter::determineSupportsWithHyperv(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div)
    {
        // boundary nodes need to be in search_among for lap/conv/div
        assert(search_among_lap_conv.size() == for_nodes_lap_conv.size());
        assert(search_among_grad.size() == for_nodes_grad.size());
        assert(search_among_div.size() == for_nodes_div.size());

        FindClosest supp_hyp(n_hyp);
        mm::Range<int> first_n_conv = mm::Range<int>::seq(n[poly_conv]);
        mm::Range<int> first_n_lap = mm::Range<int>::seq(n[poly_lap]);
        assert(n[poly_lap] <= n_hyp);
        assert(n[poly_conv] <= n[poly_lap]);
        for (size_t i = 0; i < for_nodes_lap_conv.size(); i++) {
            if (for_nodes_lap_conv[i].size() == 0)
                continue;
            supp_hyp.forNodes(for_nodes_lap_conv[i]).searchAmong(search_among_lap_conv[i]);
            d_hyp.findSupport(supp_hyp);
            for (int j : for_nodes_lap_conv[i]) {
                d_conv.support(j) = d_hyp.support(j)[first_n_conv];
                d_u.support(j) = d_hyp.support(j)[first_n_lap];
            }
        }

        FindClosest supp_grad(n[poly_grad]); // find support for grad over nodes in d_p at points in d_u
        for (size_t i = 0; i < for_nodes_grad.size(); i++) {
            if (for_nodes_grad[i].size() == 0)
                continue;
            supp_grad.forNodes(for_nodes_grad[i]).searchAmong(search_among_grad[i]);
            d_grad.findSupport(supp_grad);
        }

        FindClosest supp_divu(n[poly_div]); // find support for div u over nodes in d_u at points in d_p
        for (size_t i = 0; i < for_nodes_div.size(); i++) {
            if (for_nodes_div[i].size() == 0)
                continue;
            supp_divu.forNodes(for_nodes_div[i]).searchAmong(search_among_div[i]);
            d_div.findSupport(supp_divu);
        }
    }

    void OseenDiscretizationBetter::determineSupports_v2(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div)
    {
        // boundary nodes need to be in search_among for lap/conv/div
        assert(search_among_lap_conv.size() == for_nodes_lap_conv.size());
        assert(search_among_grad.size() == for_nodes_grad.size());
        assert(search_among_div.size() == for_nodes_div.size());

        int size_lap_conv = 0;
        for (size_t i = 0; i < for_nodes_lap_conv.size(); i++)
            size_lap_conv += for_nodes_lap_conv[i].size();

        if (size_lap_conv > 0) {
            std::vector<int> for_nodes_lap_conv_all;
            for_nodes_lap_conv_all.reserve(size_lap_conv);
            for (size_t i = 0; i < for_nodes_lap_conv.size(); i++)
                for (int j : for_nodes_lap_conv[i])
                    for_nodes_lap_conv_all.push_back(j);

            FindClosest supp_vel(3*n[poly_lap]);
            supp_vel.forNodes(for_nodes_lap_conv_all);
            d_u.findSupport(supp_vel);

            for (size_t i = 0; i < for_nodes_lap_conv.size(); i++) {
                std::unordered_set<int> set2(search_among_lap_conv[i].begin(), search_among_lap_conv[i].end());
                std::vector<int> recompute_support;
                for (int j : for_nodes_lap_conv[i]) {
                    d_u.support(j) = intersection(d_u.support(j), set2, n[poly_lap]);
                    if (d_u.support(j).size() < n[poly_conv]) { // this should be rare (source trust me)
                        recompute_support.push_back(j);
                    }
                }

                // this is expensive as kd-tree needs to be rebuilt, but should be rare
                if (recompute_support.size() > 0) {
                    supp_vel.forNodes(recompute_support).searchAmong(search_among_lap_conv[i]).numClosest(n[poly_lap]);
                    d_u.findSupport(supp_vel);
                }

                for (int j : for_nodes_lap_conv[i]) {
                    d_conv.support(j) = d_u.support(j)[Range<int>::seq(n[poly_conv])];
                }
            }
        }
        
        int size_grad = 0;
        for (size_t i = 0; i < for_nodes_grad.size(); i++)
            size_grad += for_nodes_grad[i].size();

        if (size_grad > 0) {
            std::vector<int> for_nodes_grad_all;
            for_nodes_grad_all.reserve(size_grad);
            for (size_t i = 0; i < for_nodes_grad.size(); i++)
                for (int j : for_nodes_grad[i])
                    for_nodes_grad_all.push_back(j);

            FindClosest supp_grad(4*n[poly_grad]);
            supp_grad.forNodes(for_nodes_grad_all).searchAmong(Range<int>::seq(d_u.size(), d_u.size() + d_p.size()));
            d_grad.findSupport(supp_grad);

            for (size_t i = 0; i < for_nodes_grad.size(); i++) {
                std::unordered_set<int> set2(search_among_grad[i].begin(), search_among_grad[i].end());
                std::vector<int> recompute_support;
                for (int j : for_nodes_grad[i]) {
                    d_grad.support(j) = intersection(d_grad.support(j), set2, n[poly_grad]);
                    // this is rare but can happen, supports slightly smaller than n[poly_grad] are no issue
                    if (d_grad.support(j).size() < 0.6*n[poly_grad]) { 
                        recompute_support.push_back(j);
                    }
                }

                if (recompute_support.size() > 0) {
                    supp_grad.forNodes(recompute_support).searchAmong(search_among_grad[i]).numClosest(n[poly_grad]);
                    d_grad.findSupport(supp_grad);
                }
            }
        }
        
        int size_div = 0;
        for (size_t i = 0; i < for_nodes_div.size(); i++)
            size_div += for_nodes_div[i].size();

        if (size_div > 0) {
            std::vector<int> for_nodes_div_all;
            for_nodes_div_all.reserve(size_div);
            for (size_t i = 0; i < for_nodes_div.size(); i++)
                for (int j : for_nodes_div[i])
                    for_nodes_div_all.push_back(j);

            FindClosest supp_div(3*n[poly_div]);
            supp_div.forNodes(for_nodes_div_all).searchAmong(Range<int>::seq(d_u.size()));
            d_div.findSupport(supp_div);

            for (size_t i = 0; i < for_nodes_div.size(); i++) {
                std::unordered_set<int> set2(search_among_div[i].begin(), search_among_div[i].end());
                std::vector<int> recompute_support;
                for (int j : for_nodes_div[i]) {
                    d_div.support(j) = intersection(d_div.support(j), set2, n[poly_div]);
                    // this is rare but can happen, supports slightly smaller than n[poly_grad] are no issue
                    if (d_div.support(j).size() < 0.6*n[poly_div]) { 
                        recompute_support.push_back(j);
                    }
                }

                if (recompute_support.size() > 0) {
                    supp_div.forNodes(recompute_support).searchAmong(search_among_div[i]).numClosest(n[poly_div]);
                    d_u.findSupport(supp_div);
                }
            }
        }
    }

    void OseenDiscretizationBetter::determineSupports(const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div)
    {
        // indexes in search_among_lap_conv/div need to be offset by d_u.N_ub
        assert(search_among_grad.size() == for_nodes_grad.size());
        assert(search_among_div.size() == for_nodes_div.size());

        FindClosest supp_grad(n[poly_grad]); // find support for grad over nodes in d_p at points in d_u
        for (size_t i = 0; i < for_nodes_grad.size(); i++) {
            supp_grad.forNodes(for_nodes_grad[i]).searchAmong(search_among_grad[i]);
            d_grad.findSupport(supp_grad);
        }

        FindClosest supp_divu(n[poly_div]); // find support for div u over nodes in d_u at points in d_p
        for (size_t i = 0; i < for_nodes_div.size(); i++) {
            supp_divu.forNodes(for_nodes_div[i]).searchAmong(search_among_div[i]);
            d_div.findSupport(supp_divu);
        }
    }

    void OseenDiscretizationBetter::setHyperViscosity(bool use_hyperviscosity)
    {
        this->use_hyperviscosity = use_hyperviscosity;
        if (use_hyperviscosity) {
            poly_hyp = std::floor(1.5*std::log(n[poly_conv])); // polynomial augmentation degree for hyperviscosity
            k_hyp = 2*poly_hyp+1; // degree of polyharmonic spline for hyperviscosity
            n_hyp = 2*binom(poly_hyp+d_u.dim, d_u.dim) + 1; // stencil size for hyperviscosity
        } else {
            poly_hyp = 0;
            k_hyp = 0;
            n_hyp = 0;
        }
    }

    void OseenDiscretizationBetter::determineSupportsHyperViscosity()
    {
        assert(n_hyp > 0);
        assert(use_hyperviscosity);
        FindClosest supp_hyp(n_hyp);
        supp_hyp.forNodes(idxs_ui); // don't need stencils at boundary
        d_hyp.findSupport(supp_hyp); // find support for lap within d_u
    }

    void OseenDiscretizationBetter::determineSupportsHyperViscosity(const std::vector<std::vector<int>>& search_among_hyp, 
        const std::vector<std::vector<int>>& for_nodes_hyp)
    {
        FindClosest supp_hyp(n_hyp);
        for (size_t i = 0; i < for_nodes_hyp.size(); i++) {
            supp_hyp.forNodes(for_nodes_hyp[i]).searchAmong(search_among_hyp[i]);
            d_hyp.findSupport(supp_hyp);
        }
    }

    void OseenDiscretizationBetter::computeSubset(double s) {
        assert(d_u_int.size() > 0);
        mm::DomainDiscretization<mm::Vec3d> d_temp = d_u_int;
        mm::Range<mm::Vec3d> pos_fine = d_temp.positions();
        // sort by first dimension and store permutation
        std::vector<int> perm(pos_fine.size());
        for (std::size_t i = 0; i < perm.size(); i++)
            perm[i] = i;
        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) { return pos_fine[a][0] < pos_fine[b][0]; });
        d_temp.shuffleNodes(perm);

        int k = ceil(25*s);
        FindClosest find_closest(k);
        find_closest.forceSelf(true);
        d_temp.findSupport(find_closest);

        std::vector<int> flag(d_temp.size(), 0);
        mm::Range<int> removed_nodes;
        removed_nodes.reserve(d_temp.size());

        for (int i : d_temp.all()) {
            if (flag[i] == 0) {
                double dist_to_closest = (d_temp.supportNode(i,0)-d_temp.supportNode(i, 1)).norm();
                for (int j : d_temp.support(i)) {
                    if (j > i) {
                        double dist = (d_temp.supportNode(i,0)-d_temp.pos(j)).norm();
                        if (dist < dist_to_closest*s) {
                            flag[j] = 1;
                            removed_nodes.push_back(perm[j]);
                        }
                    }
                }
            }
        }

        d_p = d_u_int;
        d_p.removeNodes(removed_nodes);
    }

    void OseenDiscretizationBetter::initialize(const PolyhedronShape<Vec3d>& shape, bool neumann, double s, bool subset) {
        // create point set for Laplacian of velocity
        d_u = shape.discretizeBoundaryWithStep(dx_u);
        // determine dirichlet and neumann boundary nodes indices
		double define_neumann_bnd = 1.75;
		if (neumann == true)
			define_neumann_bnd = 0.75;

		auto determine_neumann = [&define_neumann_bnd](const Vec3d& p) { return p[0] >= define_neumann_bnd; };

        idxs_dir = d_u.positions().filter(std::not_fn(determine_neumann));
        idxs_neu = d_u.positions().filter(determine_neumann);

        // shuffle nodes to get all dirichlet nodes first then all neumann nodes
        Range<int> permutation = idxs_dir;
        permutation.append(idxs_neu);
        d_u.shuffleNodes(permutation);

        idxs_dir = d_u.positions().filter(std::not_fn(determine_neumann)); // update dirichlet nodes after shuffling
        idxs_neu = d_u.positions().filter(determine_neumann); // update neumann nodes after shuffling

        if (!subset)
            d_p = d_u;

        GeneralFill<Vec3d> fill;
        fill.seed(seed);
        d_u.fill(fill,dx_u); // generate nodes

        idxs_u_ghost = d_u.addGhostNodes(dx_u, -3, idxs_neu); // add ghost nodes at neumann boundary as boundary nodes (type = -3)
        idxs_u_ghost_global = d_u.types().filter([](int type) { return type == -3; }); // get global indices of ghost nodes

        d_u_int = d_u;
        d_u_int.removeNodes(idxs_dir); // includes interior nodes, neumann nodes and ghost nodes

        if (subset) {
            if (s == 1) {
                // set pressure nodes equal to interior velocity nodes
                d_p = d_u_int;
            }  
            else {
                // set pressure nodes as subset of velocity nodes
                computeSubset(s);
            }
        } else {
            // create pressure point set independently of velocity node set
            Range<int> bd = d_p.boundary(); // save initial boundary node indices
            // dx_p/2.5 can be changed to place ghost nodes closer or further away from boundary
            // respectively pressure nodes will be further or closer to the boundary
            double dx_p = s * dx_u;
            d_p.addGhostNodes(dx_p/2.5,-2); // add temporary ghost nodes used as temporary boundary nodes to fill the interior
            d_p.addGhostNodes(dx_p/2.5, 2, idxs_neu); // actual ghost nodes, added as interior nodes (type = 2)
            d_p.removeNodes(bd); // remove initial boundary nodes
            d_p.fill(fill,dx_p); // fill domain defined by tmeporary boundary nodes
            d_p.removeBoundaryNodes(); // remove temporary ghost nodes
        }

        // laplacian and convection use same point set, but separate them to use different stencils
        d_conv = d_u;
        d_hyp = d_u;
        // create combined point sets for gradient and divergence and a set of interior velocity nodes
        d_grad = d_u;
        idxs_p = d_grad.addNodes(d_p);
        idxs_u = d_u.all();
        // neumann nodes are boundary nodes technically, so a normal is computed,
        // but they are afterwards treated as interior nodes such that the equation is enforced at the neumann nodes
        idxs_ui = idxs_neu; idxs_ui.append(d_u.interior()); 
        d_div = d_grad;

        N_u = d_u.size();
        N_ui = idxs_ui.size(); // #interior nodes + #neumann boundary nodes
        N_ub = idxs_dir.size(); // #dirichlet nodes
        N_uneu = idxs_neu.size(); // #neumann nodes = #ghost nodes
        N_p = d_p.size(); // #pressure nodes (interior and ghost)
        N_dofs = 3*(N_ui+N_uneu)+N_p+1;
        N_all = 3*N_u+N_p+1;

        mm::Range<mm::Vec3d> temp(N_u,0);
        conv_vec3d = temp;
        dir_bnd_vec3d = temp;
        neu_bnd_vec3d = temp;
        rhs_vec3d = temp;

        constraint = Eigen::VectorXd::Ones(N_p);
        exact_solution = Eigen::VectorXd::Zero(N_dofs);
    }

    void OseenDiscretizationBetter::initialize(const PolyhedronShape<Vec3d>& shape, std::function<bool(const Vec3d&)> is_neumann, 
            double s, bool subset, std::function<double(const Vec3d&)> dx_u_func) {
        // create point set for Laplacian of velocity
        // d_u = shape.discretizeBoundaryWithStep(dx_u);
        d_u = shape.discretizeBoundaryWithDensity(dx_u_func);

        idxs_dir = d_u.positions().filter(std::not_fn(is_neumann));
        idxs_neu = d_u.positions().filter(is_neumann);

        // shuffle nodes to get all dirichlet nodes first then all neumann nodes
        Range<int> permutation = idxs_dir;
        permutation.append(idxs_neu);
        d_u.shuffleNodes(permutation);

        idxs_dir = d_u.positions().filter(std::not_fn(is_neumann)); // update dirichlet nodes after shuffling
        idxs_neu = d_u.positions().filter(is_neumann); // update neumann nodes after shuffling

        if (!subset)
            d_p = d_u;

        GeneralFill<Vec3d> fill;
        fill.seed(seed);
        // d_u.fill(fill, dx_u); // generate nodes
        d_u.fill(fill, dx_u_func); // generate nodes

        idxs_u_ghost = d_u.addGhostNodes(dx_u_func, -3, idxs_neu); // add ghost nodes at neumann boundary as boundary nodes (type = -3)
        idxs_u_ghost_global = d_u.types().filter([](int type) { return type == -3; }); // get global indices of ghost nodes

        d_u_int = d_u;
        d_u_int.removeNodes(idxs_dir); // includes interior nodes, neumann nodes and ghost nodes

        if (subset) {
            if (s == 1) {
                // set pressure nodes equal to interior velocity nodes
                d_p = d_u_int;
            }  
            else {
                // set pressure nodes as subset of velocity nodes
                computeSubset(s);
            }
        } else {
            // create pressure point set independently of velocity node set
            Range<int> bd = d_p.boundary(); // save initial boundary node indices
            // dx_p/2.5 can be changed to place ghost nodes closer or further away from boundary
            // respectively pressure nodes will be further or closer to the boundary
            // double dx_p = s * dx_u;
            auto dx_p_func = [s, dx_u_func](const Vec3d& p) { return s * dx_u_func(p); };
            auto dx_p_func_2 = [s, dx_u_func](const Vec3d& p) { return s * dx_u_func(p) * 0.5; };
            d_p.addGhostNodes(dx_p_func_2, -2); // add temporary ghost nodes used as temporary boundary nodes to fill the interior
            d_p.addGhostNodes(dx_p_func_2, 2, idxs_neu); // actual ghost nodes, added as interior nodes (type = 2)
            d_p.removeNodes(bd); // remove initial boundary nodes
            d_p.fill(fill, dx_p_func); // fill domain defined by tmeporary boundary nodes
            d_p.removeBoundaryNodes(); // remove temporary ghost nodes
        }

        // laplacian and convection use same point set, but separate them to use different stencils
        d_conv = d_u;
        d_hyp = d_u;
        // create combined point sets for gradient and divergence and a set of interior velocity nodes
        d_grad = d_u;
        idxs_p = d_grad.addNodes(d_p);
        idxs_u = d_u.all();
        // neumann nodes are boundary nodes technically, so a normal is computed,
        // but they are afterwards treated as interior nodes such that the equation is enforced at the neumann nodes
        idxs_ui = idxs_neu; idxs_ui.append(d_u.interior()); 
        d_div = d_grad;

        N_u = d_u.size();
        N_ui = idxs_ui.size(); // #interior nodes + #neumann boundary nodes
        N_ub = idxs_dir.size(); // #dirichlet nodes
        N_uneu = idxs_neu.size(); // #neumann nodes = #ghost nodes
        N_p = d_p.size(); // #pressure nodes (interior and ghost)
        N_dofs = 3*(N_ui+N_uneu)+N_p+1;
        N_all = 3*N_u+N_p+1;

        mm::Range<mm::Vec3d> temp(N_u,0);
        conv_vec3d = temp;
        dir_bnd_vec3d = temp;
        neu_bnd_vec3d = temp;
        rhs_vec3d = temp;

        constraint = Eigen::VectorXd::Ones(N_p);
        exact_solution = Eigen::VectorXd::Zero(N_dofs);
    }

    void OseenDiscretizationBetter::setConstraint(std::string OFF_file, int max_order)
    {
        constraint = compute_pressure_constraint(OFF_file, d_p, max_order);
    }

    void OseenDiscretizationBetter::setConvection(int which_convection)
    {
        switch (which_convection) {
            // simple convection in x-direction
            case 0:
                for (int i : idxs_ui)
                    conv_vec3d[i] = {1.0, 0.0, 0.0};
                break;
            // more complicated convection
            case 1: {
                double scaling = 6.40936351829;
                for (int i : idxs_ui) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    conv_vec3d[i] = {scaling*2.0*x*(1.0-x)*(2.0*y-1)*z, -scaling*(2.0*x-1)*y*(1.0-y), -scaling*(2.0*x-1)*(2.0*y-1.0)*z*(1.0-z)};
                }
                }
                break;
            // no convection (Stokes equations)
            case 2:
                for (int i : idxs_ui)
                    conv_vec3d[i] = {0.0, 0.0, 0.0};
                break;
            
            default:
                break;
        }
    }

    void OseenDiscretizationBetter::setBoundary(int which_dir_bnd, DomainGeometry domain_geometry)
    {
        switch (which_dir_bnd) {
            case 0:
                for (int i : idxs_dir)
                    dir_bnd_vec3d[i] = {0.0, 0.0, 0.0};
                for (int i : idxs_neu) {
                    double value = 0;
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944;
                    } else {
                        value = std::sin(PI*x)*std::cos(PI*y);
                    }
                    Vec3d normal = d_u.normal(i);
                    neu_bnd_vec3d[i] = {-normal[0]*value, -normal[1]*value, -normal[2]*value};
                }
                break;
            case 1:
                for (int i : idxs_dir)
                    dir_bnd_vec3d[i] = {1.0, 0.0, 0.0};
                for (int i : idxs_neu) {
                    double value = 0;
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944;
                    } else {
                        value = std::sin(PI*x)*std::cos(PI*y);
                    }
                    Vec3d normal = d_u.normal(i);
                    neu_bnd_vec3d[i] = {-normal[0]*value, -normal[1]*value, -normal[2]*value};
                }
                break;
            case 2:
                for (int i : idxs_dir)
                    dir_bnd_vec3d[i] = {1.0, 1.0, 1.0};
                for (int i : idxs_neu) {
                    double value = 0;
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944;
                    } else {
                        value = std::sin(PI*x)*std::cos(PI*y);
                    }
                    Vec3d normal = d_u.normal(i);
                    neu_bnd_vec3d[i] = {-normal[0]*value, -normal[1]*value, -normal[2]*value};
                }
                break;
            
            case 3:
                for (int i : idxs_dir) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    dir_bnd_vec3d[i] = {cos(x)*sin(y), x*cos(y), z*(x+sin(x))*sin(y)};
                }
                for (int i : idxs_neu) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    double value = 0, der1x = 0, der1y = 0, der1z = 0;
                    Vec3d normal = d_u.normal(i);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944;
                    } else {
                        value = std::sin(PI*x)*std::cos(PI*y);
                    }
                    der1x = -normal[0]*sin(x)*sin(y) + normal[1]*cos(x)*cos(y);
                    der1y = normal[0]*cos(y) - normal[1]*x*sin(y);
                    der1z = normal[0]*z*(cos(x)+1)*sin(y) + normal[1]*z*(x+sin(x))*cos(y) + normal[2]*(x+sin(x))*sin(y);
                    neu_bnd_vec3d[i] = {nu*der1x - normal[0]*value, nu*der1y - normal[1]*value, nu*der1z - normal[2]*value};
                }
                break;

            case 4:
                for (int i : idxs_dir)
                    dir_bnd_vec3d[i] = {1.0, 1.0, 1.0};
                for (int i : idxs_neu) {
                    double value = 0;
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = (x-0.5)*(y-0.5)*(z-0.5) - 0.001363726470335;
                    } else {
                        value = (x-0.5)*(y-0.5)*(z-0.5);
                    }
                    Vec3d normal = d_u.normal(i);
                    neu_bnd_vec3d[i] = {-normal[0]*value, -normal[1]*value, -normal[2]*value};
                }
                break;

            case 5:
                for (int i : idxs_dir) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    dir_bnd_vec3d[i] = {cos(x)*sin(y), x*cos(y), z*(x+sin(x))*sin(y)};
                }
                for (int i : idxs_neu) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    double value = 0, der1x = 0, der1y = 0, der1z = 0;
                    Vec3d normal = d_u.normal(i);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        value = (x-0.5)*(y-0.5)*(z-0.5) - 0.001363726470335;
                    } else {
                        value = (x-0.5)*(y-0.5)*(z-0.5);
                    }
                    der1x = -normal[0]*sin(x)*sin(y) + normal[1]*cos(x)*cos(y);
                    der1y = normal[0]*cos(y) - normal[1]*x*sin(y);
                    der1z = normal[0]*z*(cos(x)+1)*sin(y) + normal[1]*z*(x+sin(x))*cos(y) + normal[2]*(x+sin(x))*sin(y);
                    neu_bnd_vec3d[i] = {nu*der1x - normal[0]*value, nu*der1y - normal[1]*value, nu*der1z - normal[2]*value};
                }
                break;
            // benchmark
            case 6:
                for (int i : idxs_dir) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    double H = 0.41;
                    double U = 0.45;
                    if (x == 0) 
                        dir_bnd_vec3d[i] = {16*U*y*z*(H-y)*(H-z)/std::pow(H, 4), 0.0, 0.0};
                    else
                        dir_bnd_vec3d[i] = {0.0, 0.0, 0.0};
                }
                for (int i : idxs_neu) {
                    neu_bnd_vec3d[i] = {0.0, 0.0, 0.0};
                }
                break;
            
            default:
                break;
        }
    }

    // set the right hand side according to the chosen exact solution for interior nodes
    void OseenDiscretizationBetter::setRHS(int which_rhs)
    {
        switch (which_rhs) {
            case 0:
            case 1:
            case 2:
                for (int i : idxs_ui) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    rhs_vec3d[i] = {PI*cos(PI*x)*cos(PI*y),
                                        -PI*sin(PI*x)*sin(PI*y),
                                        0.0};
                }
                break;
            case 3:
                for (int i : idxs_ui) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    rhs_vec3d[i] = {2.0*nu*cos(x)*sin(y)-conv_vec3d[i](0)*sin(x)*sin(y)+conv_vec3d[i](1)*cos(x)*cos(y) + PI*cos(PI*x)*cos(PI*y),
                                        nu*x*cos(y)+conv_vec3d[i](0)*cos(y)-conv_vec3d[i](1)*x*sin(y) - PI*sin(PI*x)*sin(PI*y),
                                        nu*z*(x+2.0*sin(x))*sin(y)+conv_vec3d[i](0)*z*(cos(x)+1)*sin(y)+conv_vec3d[i](1)*z*(x+sin(x))*cos(y)+conv_vec3d[i](2)*(x+sin(x))*sin(y)};
                }
                break;

            case 4:
                for (int i : idxs_ui) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    rhs_vec3d[i] = {(y-0.5)*(z-0.5), (x-0.5)*(z-0.5), (x-0.5)*(y-0.5)};
                }
                break;

            case 5:
                for (int i : idxs_ui) {
                    double x = d_u.pos(i,0);
                    double y = d_u.pos(i,1);
                    double z = d_u.pos(i,2);
                    rhs_vec3d[i] = {2.0*nu*cos(x)*sin(y)-conv_vec3d[i](0)*sin(x)*sin(y)+conv_vec3d[i](1)*cos(x)*cos(y) + (y-0.5)*(z-0.5),
                                        nu*x*cos(y)+conv_vec3d[i](0)*cos(y)-conv_vec3d[i](1)*x*sin(y) + (x-0.5)*(z-0.5),
                                        nu*z*(x+2.0*sin(x))*sin(y)+conv_vec3d[i](0)*z*(cos(x)+1)*sin(y)+conv_vec3d[i](1)*z*(x+sin(x))*cos(y)+conv_vec3d[i](2)*(x+sin(x))*sin(y) + (x-0.5)*(y-0.5)};
                }
                break;
            
            default:
                break;
        }
    }

    // set the exact solution
    void OseenDiscretizationBetter::computeExactSolution(int which_solution, DomainGeometry domain_geometry)
    {
        switch (which_solution) {
            case 0:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    exact_solution(i) = 0.0;
                    exact_solution(i+(N_ui+N_uneu)) = 0.0;
                    exact_solution(i+2*(N_ui+N_uneu)) = 0.0;
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944; // integral of this over bunny domain is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;
            case 1:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    exact_solution(i) = 1.0;
                    exact_solution(i+(N_ui+N_uneu)) = 0.0;
                    exact_solution(i+2*(N_ui+N_uneu)) = 0.0;
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944; // integral of this over bunny domain is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;
            case 2:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    exact_solution(i) = 1.0;
                    exact_solution(i+(N_ui+N_uneu)) = 1.0;
                    exact_solution(i+2*(N_ui+N_uneu)) = 1.0;
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944; // integral of this over bunny domain is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;
            case 3:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    double x = d_u_int.pos(i,0);
                    double y = d_u_int.pos(i,1);
                    double z = d_u_int.pos(i,2);
                    exact_solution(i) = cos(x)*sin(y);
                    exact_solution(i+(N_ui+N_uneu)) = x*cos(y);
                    exact_solution(i+2*(N_ui+N_uneu)) = z*(x+sin(x))*sin(y);
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y) - 0.260568484878944; // integral of this over bunny domain is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = std::sin(PI*x)*std::cos(PI*y); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;

            case 4:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    exact_solution(i) = 1.0;
                    exact_solution(i+(N_ui+N_uneu)) = 1.0;
                    exact_solution(i+2*(N_ui+N_uneu)) = 1.0;
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    double z = d_p.pos(i,2);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = (x-0.5)*(y-0.5)*(z-0.5) - 0.001363726470335; // integral of this over bunny is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = (x-0.5)*(y-0.5)*(z-0.5); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;

            case 5:
                for(int i = 0; i < N_ui+N_uneu; i++) {
                    double x = d_u_int.pos(i,0);
                    double y = d_u_int.pos(i,1);
                    double z = d_u_int.pos(i,2);
                    exact_solution(i) = cos(x)*sin(y);
                    exact_solution(i+(N_ui+N_uneu)) = x*cos(y);
                    exact_solution(i+2*(N_ui+N_uneu)) = z*(x+sin(x))*sin(y);
                }
                for(int i : d_p.all()) {
                    double x = d_p.pos(i,0);
                    double y = d_p.pos(i,1);
                    double z = d_p.pos(i,2);
                    if (domain_geometry == DomainGeometry::BUNNY) {
                        exact_solution(i+3*(N_ui+N_uneu)) = (x-0.5)*(y-0.5)*(z-0.5) - 0.001363726470335; // integral of this over bunny is zero
                    } else {
                        exact_solution(i+3*(N_ui+N_uneu)) = (x-0.5)*(y-0.5)*(z-0.5); // integral of this over cube and narrowing domain is zero
                    }
                }
                break;
            
            default:
                break;
        }
    }

    // convenience function that sets Dirichlet and Neumann boundary condition, RHS and exact solution such that they fit together
    void OseenDiscretizationBetter::setSolution(int which_solution, DomainGeometry domain_geometry)
    {
        setBoundary(which_solution, domain_geometry);
        setRHS(which_solution);
        computeExactSolution(which_solution, domain_geometry);
    }

}