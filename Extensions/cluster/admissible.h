#include "../../H2Lib/h2lib.h"

#ifndef ADMISSIBLE_HEADER
#define ADMISSIBLE_HEADER

struct adm_sparse_data
{
  uint nmax;
  psparsematrix sp;
  uint* col;
  uint* row;

  adm_sparse_data(uint nmax, psparsematrix sp, uint* col, uint* row)
    : nmax(nmax), sp(sp), col(col), row(row) {}

  ~adm_sparse_data() {
    del_sparsematrix(sp);
    delete[] col;
    delete[] row;
  }
};

struct reordered_sparsematrix
{
    psparsematrix sp; // reordered sparse matrix
    uint* col; // column permutation
    uint* row; // row permutation
};

reordered_sparsematrix 
getReorderedSparseMatrix(psparsematrix sp, uint* rowinv, uint* colinv);

/* ************************************
 * Coupled admissibility conditions   *
 **************************************/

bool
admissible_coupled_cluster(pcluster s, pcluster t, void *data);

bool
admissible_coupled_cluster_v2(pcluster rc, pcluster cc, void *data);

/****************************************
 * Weaker admissibility conditions      *
 ****************************************/

bool
admissible_hodlr(pcluster s, pcluster t, void *data);

bool 
admissible_sparse_row_col(pcluster s, pcluster t, void *data);

bool 
admissible_sparse(pcluster s, pcluster t, void *data);

bool
admissible_weak(pcluster s, pcluster t, void *data);

bool
admissible_weak_rbffd(pcluster s, pcluster t, void *data);

bool
admissible_2_min_cluster_rbffd(pcluster rc, pcluster cc, void *data);

/* *****************************************
 * DD with weaker admissibility conditions *
 *******************************************/

bool
admissible_dd_sparse(pcluster s, pcluster t, void *data);

bool
admissible_dd_sparse_row_col(pcluster s, pcluster t, void *data);

bool
admissible_dd_sparse_strict(pcluster s, pcluster t, void *data);

bool
admissible_dd_weak(pcluster s, pcluster t, void *data);

bool
admissible_dd_weak_rbffd(pcluster s, pcluster t, void *data);

bool
admissible_dd_strong(pcluster s, pcluster t, void *data);

/* **********************************************
 * Coupled with weaker admissibility conditions *
 ************************************************/

bool
admissible_coupled_sparse(pcluster s, pcluster t, void *data);

bool
admissible_coupled_weak(pcluster s, pcluster t, void *data);

bool
admissible_coupled_hodlr(pcluster s, pcluster t, void *data);

/* *****************************************
 * IA admissibility conditions *
 *******************************************/

bool
admissible_ia_cluster(pcluster s, pcluster t, void *data);

bool
admissible_ia_sparse(pcluster s, pcluster t, void *data);

bool
admissible_ia_weak(pcluster s, pcluster t, void *data);

/* *****************************************
 * IA coupled admissibility conditions *
 *******************************************/

bool
admissible_coupled_ia_cluster(pcluster s, pcluster t, void *data);

bool
admissible_coupled_ia_sparse(pcluster s, pcluster t, void *data);

bool
admissible_coupled_ia_weak(pcluster s, pcluster t, void *data);

#endif