
#include "H2Lib/h2lib.h"

/* Include extended modules */
#include "Extensions/preconditioning/prec.h"
#include "Extensions/preconditioning/block_diagonal.h"
#include "Extensions/preconditioning/hlu.h"

#include "Extensions/sparse/spmatrix.h"
#include "Extensions/sparse/sparse_compression.h"

#include "Extensions/cluster/oseencluster.h"
#include "Extensions/cluster/admissible.h"

#include "Extensions/solver/solver.h"
#include "Extensions/solver/bicgstab.h"
#include "Extensions/solver/gmres.h"

#include "Extensions/auxiliaries/aux_h2lib.h"
#include "Extensions/auxiliaries/aux_medusa.hpp"
#include "Extensions/auxiliaries/aux_eigen.hpp"
#include "Extensions/auxiliaries/polyhedron_integration.h"
#include "Extensions/auxiliaries/spmatrix_medusa.h"

#include "Extensions/discretization/weights.hpp"
#include "Extensions/discretization/domain.hpp"
#include "Extensions/discretization/support_by_cluster.hpp"

#include "Extensions/io/io.h"

#include "Extensions/harith/harith3.hpp"
#include "Extensions/harith/lanczostrunc.hpp"
#include "Extensions/harith/rand_trunc.hpp"
#include "Extensions/harith/sumexpression.hpp"
#include "Extensions/harith/transpose.hpp"
#include "Extensions/harith/truncation.hpp"