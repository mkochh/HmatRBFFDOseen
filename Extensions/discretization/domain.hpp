#ifndef RBFFD_DISCRETIZATION_HEADER
#define RBFFD_DISCRETIZATION_HEADER

// #include "../../medusa/include/medusa/Medusa.hpp"
#include <medusa/Medusa_fwd.hpp>
#include "../../medusa/include/medusa/bits/domains/PolyhedronShape.hpp"
#include "../auxiliaries/polyhedron_integration.h"
#include <Eigen/Sparse>
#include "../../H2Lib/Library/cluster.h"

enum class PressureConstraint {
	POLY_QUAD, // Glaubitz
	SET, // set one pressure node to specific value
	AVEARGE // all quadrature weights are 1
};

enum class DomainGeometry {
	CUBE,
	BUNNY
};

struct RBFFDOptions {
    // domain discretization options
    DomainGeometry domain_geometry = DomainGeometry::CUBE; // geometry of the domain
    double dx_u = 0.05; // separation distance for velocity nodes
    double step_size_scale = 1.4; // dx_p = step_size_scale * dx_u
	bool subset = false; // determines whether pressure nodes are subset of velocity nodes
    int seed = 0; // seed for random number generator

    // polynomial degrees
    int poly_lap = 4; // polynomial degree for laplace stencils
    int poly_conv = 3; // polynomial degree for convection stencils
    int poly_grad = 3; // polynomial degree for gradient stencils
    int poly_div = 3; // polynomial degree for divergence stencils

    // pressure constraint
    PressureConstraint pressure_constraint = PressureConstraint::POLY_QUAD;

    // hyperviscosity
    bool use_hyperviscosity = false;

    // Solution
    double nu = 1e-2; // viscosity
    int conv = 0; // convection term see domain.hpp, domain.cpp
    int sol = 3; // manufactured solution see domain.hpp, domain.cpp
    bool neumann = false; // if true, neumann boundary conditions are used on right side of cube or bunny
};

struct Indices {
	mm::Range<int> idxs_vel;
	mm::Range<int> idxs_div;
	mm::Range<int> idxs_grad;
	mm::Range<int> idxs_neu_vel;
	mm::Range<int> idxs_neu_grad;
};

struct SearchAmongTree {
    std::vector<int> search_among;
    std::vector<int> for_nodes;
    std::vector<SearchAmongTree> sons;
};

struct ForNodesTree {
    std::vector<int> for_nodes = std::vector<int>();
    std::vector<ForNodesTree> sons;
};

namespace mm {

class Dijkstra {
	public:
		Dijkstra(const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph)
		: graph(graph) 
		{
			visited.resize(graph.rows(), false);
			dist.resize(graph.rows(), std::numeric_limits<double>::infinity());
		}

		Range<int> n_nearest(int start, int n);

		Range<int> n_nearest_constrained(int start, int n, std::vector<int> flag);

		std::vector<int> explored_edges;

	private:
		Eigen::SparseMatrix<double, Eigen::RowMajor> graph;

		std::vector<bool> visited;
		std::vector<double> dist;
		std::vector<int> visited_idx;
		std::vector<int> dist_idx;

		void reset_visited();
		void reset_dist();
};

int binom(const int n, const int k);

class OseenDiscretizationBetter {
  
  	public:
		DomainDiscretization<Vec3d> d_u;
		DomainDiscretization<Vec3d> d_p;
		DomainDiscretization<Vec3d> d_grad;
		DomainDiscretization<Vec3d> d_div;
		DomainDiscretization<Vec3d> d_conv;
		DomainDiscretization<Vec3d> d_u_int;
		DomainDiscretization<Vec3d> d_hyp;
		Range<int> idxs_p;
		Range<int> idxs_u;
		Range<int> idxs_ui;
		Range<int> idxs_u_ghost;
		Range<int> idxs_u_ghost_global;
		Range<int> idxs_neu;
		Range<int> idxs_dir;
		int N_u;
		int N_ui;
		int N_ub;
		int N_uneu;
		int N_p;
		int N_dofs;
		int N_all;
		int poly_lap;
		int poly_conv;
		int poly_grad;
		int poly_div;
		int poly_hyp;
		int n_hyp;
		int k_hyp;
		double nu;
		double dx_u;
		Eigen::VectorXd constraint;
		Eigen::VectorXd exact_solution;
		mm::Range<mm::Vec3d> conv_vec3d;
		mm::Range<mm::Vec3d> dir_bnd_vec3d;
		mm::Range<mm::Vec3d> neu_bnd_vec3d;
		mm::Range<mm::Vec3d> rhs_vec3d;
		int seed;
		bool use_hyperviscosity;
		int n[6] = {0, 10, 22, 43, 74, 116}; // stencil sizes for different deg of polynomial augmentation
		int l[6] = {0, 1, 2, 3, 4, 5}; // degree of polynomial augmentation
		int k[6] = {0, 1, 3, 3, 3, 5}; // degree of polyharmonic spline for different deg of polynomial augmentation

		OseenDiscretizationBetter(const PolyhedronShape<Vec3d>& shape, const RBFFDOptions& rbffd_opt);
		
		OseenDiscretizationBetter(const PolyhedronShape<Vec3d>& shape, const RBFFDOptions& rbffd_opt,
							std::function<bool(const Vec3d&)> is_neumann, 
							std::function<double(const Vec3d&)> dx_u_func);

		void determineSupports();

		void determineSupportsGraph();

		void determineSupports(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div);
					
		void determineSupportsWithHyperv(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div);

		void determineSupportsGraph(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div);
					
		Indices determineSupportsGraph(pcluster rootv, pcluster rootp, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
        					int max_depth = -1);

		Indices determineSupportsGreedy(pcluster rootv, pcluster rootp, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat,
        					int max_depth);

		void determineSupports_v2(const std::vector<std::vector<int>>& search_among_lap_conv, 
							const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_lap_conv,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div);

		void determineSupports(const std::vector<std::vector<int>>& search_among_grad,
							const std::vector<std::vector<int>>& search_among_div,
							const std::vector<std::vector<int>>& for_nodes_grad,
							const std::vector<std::vector<int>>& for_nodes_div);

		void setConstraint(std::string OFF_file, int max_order = 8);

		void setHyperViscosity(bool use_hyperviscosity);

		void determineSupportsHyperViscosity();

		void determineSupportsHyperViscosity(const std::vector<std::vector<int>>& search_among_hyp, 
			const std::vector<std::vector<int>>& for_nodes_hyp);

  	protected:

    	void initialize(const PolyhedronShape<Vec3d>& shape, bool neumann, double s, bool subset);
		
    	void initialize(const PolyhedronShape<Vec3d>& shape, std::function<bool(const Vec3d&)> is_neumann, 
			double s, bool subset, std::function<double(const Vec3d&)> dx_u_func);

	private:

		void setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
			std::vector<int> flag, Dijkstra& dijkstra, std::vector<int> interface_idx, 
			const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

		void setSearchAmongForNodesDiv(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
			std::vector<int> flag, const KDTree<Vec3d>& tree, std::vector<int> interface_idx, 
			const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

		void setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
			std::vector<int>& flag, Dijkstra& dijkstra,
        	uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

		void setSearchAmongForNodesGrad(pcluster cv, pcluster cp, std::vector<int>& to_be_updated_all, 
			std::vector<int>& flag, const KDTree<Vec3d>& tree,
        	uint min_size, const Eigen::SparseMatrix<int, Eigen::RowMajor>& mat, int max_depth = -1);

		void setConvection(int which_convection);

		void setBoundary(int which_dir_bnd, DomainGeometry domain_geometry);

		void setRHS(int which_rhs);

		void computeExactSolution(int which_solution, DomainGeometry domain_geometry);

		void setSolution(int which_solution, DomainGeometry domain_geometry);

		void computeSubset(double s);
};

} // namespace mm

#endif