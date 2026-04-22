#include <list>
#include <assert.h>
#include "truncation.hpp"
#include "sumexpression.hpp"
#include "harith3.hpp"

void SVDTruncation::truncate_rkmatrix(prkmatrix rk)
{
  trunc_rkmatrix(this->tm, this->eps, rk);
}

void SVDTruncation::addtrunc_rkmatrix(field alpha, pcrkmatrix x, prkmatrix y)
{
  add_rkmatrix(alpha, x, this->tm, this->eps, y);
}

static void
addmultrunc_svd_nn_hmatrix_rkmatrix(field alpha, pchmatrix x, pchmatrix y, SVDTruncation &trunc, prkmatrix z)
{
  if (x->f)
  {
    uint rows = x->rc->size;
    uint cols = y->cc->size;
    uint k = UINT_MIN(x->f->rows, x->f->cols);

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    if (x->f->rows > x->f->cols)
    {
      pamatrix id = new_amatrix(x->f->cols, x->f->cols);
      identity_amatrix(id);
      copy_amatrix(false, x->f, &xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, true, y, true, id, false, &xy->B);
      del_amatrix(id);
    }
    else
    {
      identity_amatrix(&xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, true, y, true, x->f, false, &xy->B);
    }
    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else if (x->r)
  {
    uint rows = x->rc->size;
    uint cols = y->cc->size;
    uint k = x->r->k;

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    copy_amatrix(false, &x->r->A, &xy->A);
    clear_amatrix(&xy->B);
    addmul_hmatrix_amatrix_amatrix(1.0, true, y, false, &x->r->B, false, &xy->B);

    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else
  {
    if (y->f)
    {
      uint rows = x->rc->size;
      uint cols = y->cc->size;
      uint k = UINT_MIN(y->f->rows, y->f->cols);

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      if (y->f->rows > y->f->cols)
      {

        identity_amatrix(&xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, false, x, false, y->f, false, &xy->A);
      }
      else
      {

        pamatrix id = new_amatrix(y->f->rows, y->f->rows);
        identity_amatrix(id);
        copy_amatrix(true, y->f, &xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, false, x, false, id, false, &xy->A);
        del_amatrix(id);
      }
      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else if (y->r)
    {
      uint rows = x->rc->size;
      uint cols = y->cc->size;
      uint k = y->r->k;

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      copy_amatrix(false, &y->r->B, &xy->B);
      clear_amatrix(&xy->A);
      addmul_hmatrix_amatrix_amatrix(1.0, false, x, false, &y->r->A, false, &xy->A);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else
    {
      pccluster rc = x->rc;
      pccluster cc = y->cc;
      bool rsplit = x->rc != x->son[0]->rc;
      bool csplit = y->cc != y->son[0]->cc;

      phmatrix split = split_rkmatrix(z, rc, cc, rsplit, csplit, false);
      clear_hmatrix(split);
      addmul3_hmatrix(1.0, false, x, false, y, trunc, false, split);
      prkmatrix xy = merge_hmatrix_rkmatrix(split, trunc.tm, trunc.eps);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);

      del_hmatrix(split);
      del_rkmatrix(xy);
    }
  }
}

static void
addmultrunc_svd_nt_hmatrix_rkmatrix(field alpha, pchmatrix x, pchmatrix y, SVDTruncation &trunc, prkmatrix z)
{
  if (x->f)
  {
    uint rows = x->rc->size;
    uint cols = y->rc->size;
    uint k = UINT_MIN(x->f->rows, x->f->cols);

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    if (x->f->rows > x->f->cols)
    {
      pamatrix id = new_amatrix(x->f->cols, x->f->cols);
      identity_amatrix(id);
      copy_amatrix(false, x->f, &xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, false, y, true, id, false, &xy->B);
      del_amatrix(id);
    }
    else
    {
      identity_amatrix(&xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, false, y, true, x->f, false, &xy->B);
    }
    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else if (x->r)
  {
    uint rows = x->rc->size;
    uint cols = y->rc->size;
    uint k = x->r->k;

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    copy_amatrix(false, &x->r->A, &xy->A);
    clear_amatrix(&xy->B);
    addmul_hmatrix_amatrix_amatrix(1.0, false, y, false, &x->r->B, false, &xy->B);

    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else
  {
    if (y->f)
    {
      uint rows = x->rc->size;
      uint cols = y->rc->size;
      uint k = UINT_MIN(y->f->rows, y->f->cols);

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      if (y->f->rows > y->f->cols)
      {

        pamatrix id = new_amatrix(y->f->cols, y->f->cols);
        identity_amatrix(id);
        copy_amatrix(false, y->f, &xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, false, x, false, id, false, &xy->A);
        del_amatrix(id);
      }
      else
      {
        identity_amatrix(&xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, false, x, true, y->f, false, &xy->A);
      }
      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else if (y->r)
    {
      uint rows = x->rc->size;
      uint cols = y->rc->size;
      uint k = y->r->k;

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      copy_amatrix(false, &y->r->A, &xy->B);
      clear_amatrix(&xy->A);
      addmul_hmatrix_amatrix_amatrix(1.0, false, x, false, &y->r->B, false, &xy->A);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else
    {
      pccluster rc = x->rc;
      pccluster cc = y->rc;
      bool rsplit = x->rc != x->son[0]->rc;
      bool csplit = y->rc != y->son[0]->rc;

      phmatrix split = split_rkmatrix(z, rc, cc, rsplit, csplit, false);
      clear_hmatrix(split);
      addmul3_hmatrix(1.0, false, x, true, y, trunc, false, split);
      prkmatrix xy = merge_hmatrix_rkmatrix(split, trunc.tm, trunc.eps);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);

      del_hmatrix(split);
      del_rkmatrix(xy);
    }
  }
}

static void
addmultrunc_svd_tn_hmatrix_rkmatrix(field alpha, pchmatrix x, pchmatrix y, SVDTruncation &trunc, prkmatrix z)
{
  if (x->f)
  {
    uint rows = x->cc->size;
    uint cols = y->cc->size;
    uint k = UINT_MIN(x->f->rows, x->f->cols);

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    if (x->f->rows > x->f->cols)
    {
      identity_amatrix(&xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, true, y, false, x->f, false, &xy->B);
    }
    else
    {
      pamatrix id = new_amatrix(x->f->rows, x->f->rows);
      identity_amatrix(id);
      copy_amatrix(true, x->f, &xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, true, y, true, id, false, &xy->B);
      del_amatrix(id);
    }
    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else if (x->r)
  {
    uint rows = x->cc->size;
    uint cols = y->cc->size;
    uint k = x->r->k;

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    copy_amatrix(false, &x->r->B, &xy->A);
    clear_amatrix(&xy->B);
    addmul_hmatrix_amatrix_amatrix(1.0, true, y, false, &x->r->A, false, &xy->B);

    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else
  {
    if (y->f)
    {
      uint rows = x->cc->size;
      uint cols = y->cc->size;
      uint k = UINT_MIN(y->f->rows, y->f->cols);

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      if (y->f->rows > y->f->cols)
      {

        identity_amatrix(&xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, true, x, false, y->f, false, &xy->A);
      }
      else
      {

        pamatrix id = new_amatrix(y->f->rows, y->f->rows);
        identity_amatrix(id);
        copy_amatrix(true, y->f, &xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, true, x, false, id, false, &xy->A);
        del_amatrix(id);
      }
      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else if (y->r)
    {
      uint rows = x->cc->size;
      uint cols = y->cc->size;
      uint k = y->r->k;

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      copy_amatrix(false, &y->r->B, &xy->B);
      clear_amatrix(&xy->A);
      addmul_hmatrix_amatrix_amatrix(1.0, true, x, false, &y->r->A, false, &xy->A);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else
    {
      pccluster rc = x->cc;
      pccluster cc = y->cc;
      bool rsplit = x->cc != x->son[0]->cc;
      bool csplit = y->cc != y->son[0]->cc;

      phmatrix split = split_rkmatrix(z, rc, cc, rsplit, csplit, false);
      clear_hmatrix(split);
      addmul3_hmatrix(1.0, true, x, false, y, trunc, false, split);
      prkmatrix xy = merge_hmatrix_rkmatrix(split, trunc.tm, trunc.eps);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);

      del_hmatrix(split);
      del_rkmatrix(xy);
    }
  }
}

static void
addmultrunc_svd_tt_hmatrix_rkmatrix(field alpha, pchmatrix x, pchmatrix y, SVDTruncation &trunc, prkmatrix z)
{
  if (x->f)
  {
    uint rows = x->cc->size;
    uint cols = y->rc->size;
    uint k = UINT_MIN(x->f->rows, x->f->cols);

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    if (x->f->rows > x->f->cols)
    {
      identity_amatrix(&xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, false, y, false, x->f, false, &xy->B);
    }
    else
    {
      pamatrix id = new_amatrix(x->f->rows, x->f->rows);
      identity_amatrix(id);
      copy_amatrix(true, x->f, &xy->A);
      clear_amatrix(&xy->B);
      addmul_hmatrix_amatrix_amatrix(1.0, false, y, true, id, false, &xy->B);
      del_amatrix(id);
    }
    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else if (x->r)
  {
    uint rows = x->cc->size;
    uint cols = y->rc->size;
    uint k = x->r->k;

    prkmatrix xy = new_rkmatrix(rows, cols, k);

    copy_amatrix(false, &x->r->B, &xy->A);
    clear_amatrix(&xy->B);
    addmul_hmatrix_amatrix_amatrix(1.0, false, y, false, &x->r->A, false, &xy->B);

    add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
    del_rkmatrix(xy);
  }
  else
  {
    if (y->f)
    {
      uint rows = x->cc->size;
      uint cols = y->rc->size;
      uint k = UINT_MIN(y->f->rows, y->f->cols);

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      if (y->f->rows > y->f->cols)
      {
        pamatrix id = new_amatrix(y->f->cols, y->f->cols);
        identity_amatrix(id);
        copy_amatrix(false, y->f, &xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, true, x, false, id, false, &xy->A);
        del_amatrix(id);
      }
      else
      {
        identity_amatrix(&xy->B);
        clear_amatrix(&xy->A);
        addmul_hmatrix_amatrix_amatrix(1.0, true, x, true, y->f, false, &xy->A);
      }
      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else if (y->r)
    {
      uint rows = x->cc->size;
      uint cols = y->rc->size;
      uint k = y->r->k;

      prkmatrix xy = new_rkmatrix(rows, cols, k);

      copy_amatrix(false, &y->r->A, &xy->B);
      clear_amatrix(&xy->A);
      addmul_hmatrix_amatrix_amatrix(1.0, true, x, false, &y->r->B, false, &xy->A);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);
      del_rkmatrix(xy);
    }
    else
    {
      pccluster rc = x->cc;
      pccluster cc = y->rc;
      bool rsplit = x->cc != x->son[0]->cc;
      bool csplit = y->rc != y->son[0]->rc;

      phmatrix split = split_rkmatrix(z, rc, cc, rsplit, csplit, false);
      clear_hmatrix(split);
      addmul3_hmatrix(1.0, true, x, true, y, trunc, false, split);
      prkmatrix xy = merge_hmatrix_rkmatrix(split, trunc.tm, trunc.eps);

      add_rkmatrix(alpha, xy, trunc.tm, trunc.eps, z);

      del_hmatrix(split);
      del_rkmatrix(xy);
    }
  }
}

static void
addmultrunc_svd_hmatrix_rkmatrix(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y, SVDTruncation &trunc, prkmatrix z)
{
  if(xtrans)
  {
    if(ytrans)
      addmultrunc_svd_tt_hmatrix_rkmatrix(alpha, x, y, trunc, z);
    else
      addmultrunc_svd_tn_hmatrix_rkmatrix(alpha, x, y, trunc, z);
  }
  else
  {
    if(ytrans)
      addmultrunc_svd_nt_hmatrix_rkmatrix(alpha, x, y, trunc, z);
    else
      addmultrunc_svd_nn_hmatrix_rkmatrix(alpha, x, y, trunc, z);
  }
}

void SVDTruncation::addmultrunc_hmatrix_rkmatrix(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y, prkmatrix z)
{
  addmultrunc_svd_hmatrix_rkmatrix(alpha, xtrans, x, ytrans, y, *this, z);
}


void SVDTruncation::addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix z)
{
  // do not truncate if it is only one rk-matrix (the matrix itself)
  if (src.h_sum_list.size() > 0 || src.rk_sum_list.size() > 1) {
    for(HSummand &entry : src.h_sum_list)
    {
      addmultrunc_svd_hmatrix_rkmatrix(entry.alpha, entry.xtrans, entry.x, entry.ytrans, entry.y, *this, z);
    }

    for(prkmatrix entry : src.rk_sum_list)
    {
      add_rkmatrix(1.0, entry, this->tm, this->eps, z);
    }
  }
}
