#include "admissible.h"
#include "../sparse/sparse_compression.h"
#include <vector>

reordered_sparsematrix getReorderedSparseMatrix(psparsematrix sp, uint* rowinv, uint* colinv)
{
    psparsematrix rsp;
    uint *col, *row;           // Column / Row permutation

    col = new uint[sp->cols];
    row = new uint[sp->rows];

    for (uint i = 0; i < sp->cols; i++)
        col[colinv[i]] = i;
    for (uint i = 0; i < sp->rows; i++)
        row[rowinv[i]] = i;

    rsp = reorder_sparsematrix(sp, rowinv, col);

    return {rsp, col, row};
}

/* ************************************
 * Coupled admissibility conditions   *
 **************************************/

bool admissible_coupled_cluster(pcluster s, pcluster t, void *data)
{
  bool b, a;

  a = admissible_2_min_cluster(s, t, data);

  if (a == true)
    b = true;
  else
  {
    if (s->associated != nullptr) {
      a = (s->associated == t);
      if ((t->type == 1 && s->associated->type == t->type && a == false /*&& s != t */))
        b = true;
      else
        b = false;
    } else {
      b = false;
    }
  }
  return b;
}

bool admissible_coupled_cluster_v2(pcluster rc, pcluster cc, void *data)
{
  bool a = admissible_2_cluster(rc, cc, data);

  if (a == true)
  {
    return true;
  }
  else if(rc->type == 1 && cc->type == 1 && rc != cc)
  {
    return true;
  }
  else if(rc->type == 4 && cc->type == 1)
  {
    return true;
  }
  else if(rc->type == 1 && cc->type == 4)
  {
    return true;
  }
  else if(rc->type == 3 && cc->type == 1 && rc->associated != cc && rc->associated != NULL)
  {
    return true;
  }
  else if(rc->type == 1 && cc->type == 3 && cc->associated != rc && cc->associated != NULL)
  {
    return true;
  }
  else if(rc->type == 3 && cc->type == 3 && rc != cc)
  {
    return true;
  }
  else
  {
    return false;
  }
}

/****************************************
 * Weaker admissibility conditions      *
 ****************************************/

bool admissible_hodlr(pcluster s, pcluster t, void *data)
{
  (void) data;

  if(s == t)
  {
    return false;
  }

  return true;
}

bool admissible_sparse_row_col(pcluster s, pcluster t, void *data)
{
  adm_sparse_data * adm_data = *(adm_sparse_data **) data;
  uint nmax = adm_data->nmax;
  psparsematrix sp = adm_data->sp; // assumes sp is ordered according to the cluster ordering
  uint* col = adm_data->col;
  uint* row = adm_data->row;

  // approximate row and column rank by counting non-zero rows and columns respectively
  // s is the row cluster, t is the column cluster

  if(s == t)
  {
    return false;
  }

  uint roff = row[s->idx[0]];
  uint coff = col[t->idx[0]];
  uint rows = s->size;
  uint cols = t->size;

  uint col_rank = 0;
  uint row_rank = 0;

  std::vector<int> col_flag(t->size, 0); // to keep track of unique column indices

  for (uint i = roff; i < roff + rows; i++)
  {
      uint start = sp->row[i];
      uint end = sp->row[i + 1];

      bool nzr = true;

      for (uint k = start; k < end; k++)
      {
          uint j = sp->col[k];
          // Assume ascending order of the column indices
          if (j < coff)
            continue;
          else if (j >= coff && j < coff + cols) {
            if(nzr) {
              row_rank++;
              nzr = false;
            }
            col_flag[j - coff] = 1; // Store unique column indexes in ascending order
          }
          else if (j >= coff + cols)
              break;
      }
  }

  for (int i : col_flag) {
    col_rank += i;
  }

  uint rank = std::min(row_rank, col_rank);

  // if it is more efficient to store low rank representation and rank estimation smaller than maximal rank use low rank representation
  if(rank < (s->size * t->size)/(2*(s->size + t->size)) && rank <= nmax)
  {
    return true;
  }

  return false;
}

bool admissible_sparse(pcluster s, pcluster t, void *data)
{
  adm_sparse_data * adm_data = *(adm_sparse_data **) data;
  uint nmax = adm_data->nmax;
  psparsematrix sp = adm_data->sp; // assumes sp is ordered according to the cluster ordering
  uint* col = adm_data->col;
  uint* row = adm_data->row;

  if(s == t)
  {
    return false;
  }

  // count non-zero entries in the block sp(s,t)

  uint roff = row[s->idx[0]];
  uint coff = col[t->idx[0]];
  uint rows = s->size;
  uint cols = t->size;

  uint nnz = 0; // number of non-zero entries in the block

  for (uint i = roff; i < roff + rows; i++)
  {
      uint start = sp->row[i];
      uint end = sp->row[i + 1];

      for (uint k = start; k < end; k++)
      {
          uint j = sp->col[k];
          // Assume ascending order of the column indices
          if (j < coff)
            continue;
          else if (j >= coff && j < coff + cols) {
            nnz++;
          }
          else if (j >= coff + cols)
              break;
      }
  }

  // if it is more efficient to store low rank representation and rank estimation smaller than maximal rank use low rank representation
  if (nnz < (s->size * t->size)/(2*(s->size + t->size)) && nnz <= nmax) {
    return true;
  }

  return false;
}

bool admissible_weak(pcluster s, pcluster t, void *data)
{
  real dist;

  (void) data;

  dist = getdist_2_cluster(s, t);

  return (dist > 0);
}

bool admissible_weak_rbffd(pcluster s, pcluster t, void *data)
{
  real dist;
  real rad = *(real *) data;

  dist = getdist_2_cluster(s, t);

  return (dist > rad);
}

// admissible_2_min_cluster but less instead of less equal
bool admissible_2_min_cluster_rbffd(pcluster rc, pcluster cc, void *data)
{
  real      eta = *(real *) data;
  real      diamt, diams, dist;

  diamt = getdiam_2_cluster(rc);
  diams = getdiam_2_cluster(cc);
  dist = getdist_2_cluster(rc, cc);

  return (diamt < eta * dist || diams < eta * dist);
}

/* *****************************************
 * DD with weaker admissibility conditions *
 *******************************************/

bool admissible_dd_sparse(pcluster s, pcluster t, void *data)
{
  bool a;

  a = (s == t);
  if((s->type == 1) && (s->type == t->type) && (a == false))
    return true;

  a = admissible_sparse(s, t, data);

  if(a == true)
    return true;
  
  return false;

}

bool admissible_dd_sparse_strict(pcluster s, pcluster t, void *data)
{
  bool a;

  if (std::max(s->size, t->size) > 0 && (s->type == 2 && t->type == 2))
    return false;

  a = (s == t);
  if((s->type == 1) && (s->type == t->type) && (a == false))
    return true;

  a = admissible_sparse(s, t, data);

  if(a == true)
    return true;
  
  return false;

}

bool admissible_dd_sparse_row_col(pcluster s, pcluster t, void *data)
{
  bool a;

  a = (s == t);
  if((s->type == 1) && (s->type == t->type) && (a == false))
    return true;
  
  a = admissible_sparse_row_col(s, t, data);

  if(a == true)
    return true;
  
  return false;

}

bool admissible_dd_weak(pcluster s, pcluster t, void *data)
{
  bool a;

  a = admissible_weak(s, t, data);

  if(a == true)
  {
    return true;
  }
  else{
    a = (s == t);
    if((s->type == 1) && (s->type == t->type) && (a == false))
    {
      return true;
    }
    else
    {
      return false;
    }
  }
}

bool admissible_dd_weak_rbffd(pcluster s, pcluster t, void *data)
{
  bool a;

  a = admissible_weak_rbffd(s, t, data);

  if(a == true)
  {
    return true;
  }
  else{
    a = (s == t);
    if((s->type == 1) && (s->type == t->type) && (a == false))
    {
      return true;
    }
    else
    {
      return false;
    }
  }
}

bool admissible_dd_strong(pcluster s, pcluster t, void *data)
{
  bool a;

  a = (s == t);
  if((s->type == 1) && (s->type == t->type) && (a == false))
    return true;

  a = admissible_2_min_cluster_rbffd(s, t, data);

  if(a == true)
    return true;
  
  return false;

}

/* **********************************************
 * Coupled with weaker admissibility conditions *
 ************************************************/

bool admissible_coupled_sparse(pcluster s, pcluster t, void *data)
{
  bool b, a;

  assert(t->type > 0);

  a = admissible_sparse(s, t, data);

  if (a == true)
    b = true;
  else
  {
    a = (s->associated == t);
    b = (s == t->associated);

    assert(a == b);
    if ((t->type == 1 && s->associated->type == t->type && a == false))
      b = true;
    else
      b = false;
  }
  return b;
}

bool admissible_coupled_weak(pcluster s, pcluster t, void *data)
{
  bool b, a;

  assert(t->type > 0);

  (void) data;

  a = admissible_weak(s, t, 0);

  if (a == true)
    b = true;
  else
  {
    a = (s->associated == t);
    b = (s == t->associated);

    assert(a == b);
    if ((t->type == 1 && s->associated->type == t->type && a == false))
      b = true;
    else
      b = false;
  }
  return b;
}

bool admissible_coupled_hodlr(pcluster s, pcluster t, void *data)
{
  bool b, a;

  assert(t->type > 0);

  (void) data;
  a = admissible_hodlr(s->associated, t, 0);
  b = admissible_hodlr(s, t->associated, 0);

  if ((a == true) || (b == true))
    return true;
  else
  {
    a = (s->associated == t);
    b = (s == t->associated);

    assert(a == b);
    if ((t->type == 1 && s->associated->type == t->type && a == false))
      return true;
    else
      return false;
  }
}

/* *****************************************
 * IA admissibility conditions *
 *******************************************/

bool admissible_ia_cluster(pcluster s, pcluster t, void *data)
{
    bool a = admissible_dd_cluster(s, t, data);
    bool b;

    if (a == true)
        b = true;
    else if (s->type == 2 && t->type == 1)
    {
        a = (s->left == t) ||
            (s->right == t) ||
            (s == t);

        if (a == false)
            b = true;
        else
            b = false;
    }
    else if(t->type == 2 && s->type == 1)
    {
        a = (t->left == s) ||
            (t->right == s) ||
            (s == t);

        b = a ? false : true;
    }
    else
    {
        b = false;
    }

    return b;
}

bool admissible_ia_sparse(pcluster s, pcluster t, void *data)
{
    bool a = admissible_dd_sparse(s, t, data);
    bool b;

    if (a == true)
        b = true;
    else if (s->type == 2 && t->type == 1)
    {
        a = (s->left == t) ||
            (s->right == t) ||
            (s == t);

        if (a == false)
            b = true;
        else
            b = false;
    }
    else if(t->type == 2 && s->type == 1)
    {
        a = (t->left == s) ||
            (t->right == s) ||
            (s == t);

        b = a ? false : true;
    }
    else
    {
        b = false;
    }

    return b;
}

bool admissible_ia_weak(pcluster s, pcluster t, void *data)
{
    bool a = admissible_dd_weak(s, t, data);
    bool b;

    if (a == true)
        b = true;
    else if (s->type == 2 && t->type == 1)
    {
        a = (s->left == t) ||
            (s->right == t) ||
            (s == t);

        if (a == false)
            b = true;
        else
            b = false;
    }
    else if(t->type == 2 && s->type == 1)
    {
        a = (t->left == s) ||
            (t->right == s) ||
            (s == t);

        b = a ? false : true;
    }
    else
    {
        b = false;
    }

    return b;
}

/* *****************************************
 * IA coupled admissibility conditions *
 *******************************************/

bool admissible_coupled_ia_cluster(pcluster s, pcluster t, void *data)
{
    bool b, a;

    a = admissible_2_min_cluster(s, t, data);

    if (a == true)
        b = true;
    else
    {
        b = admissible_ia_cluster(s->associated, t, data);
    }
    return b;
}

bool admissible_coupled_ia_sparse(pcluster s, pcluster t, void *data)
{
    bool b, a;

    a = admissible_sparse(s, t, data);

    if (a == true)
        b = true;
    else
    {
        b = admissible_ia_sparse(s->associated, t, data);
    }
    return b;
}

bool admissible_coupled_ia_weak(pcluster s, pcluster t, void *data)
{
    bool b, a;

    a = admissible_weak(s, t, data);

    if (a == true)
        b = true;
    else
    {
        b = admissible_ia_weak(s->associated, t, data);
    }
    return b;
}
