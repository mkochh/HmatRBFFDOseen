/**
 * @file harith3.cpp
 * @author Jonas Grams (jonas.grams@tuhh.de)
 * @brief Implementation of harith3.hpp
 * @version 0.2
 * @date 2024-05-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <assert.h>
#include "harith3.hpp"
#include <vector>
#include <memory>

void addmul_hmatrix_sumexpression(SumExpression &S, TruncationOperator &trunc, phmatrix z)
{

  assert(S.rc == z->rc);
  assert(S.cc == z->cc);

  // If Z has sons, restrict sum-expression and complete the computation on the sons
  if (z->son)
  {
    for (uint i = 0; i < z->rsons; i++)
      for (uint j = 0; j < z->csons; j++)
      {
        std::unique_ptr<SumExpression> son = S.restrict_to(i, j);

        phmatrix subz = z->son[i + j * z->rsons];
        addmul_hmatrix_sumexpression(*son, trunc, subz);
      }
  }
  else if (z->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(S, z->r);
  }
  else
  {
    assert(z->f);
    S.add_amatrix(z->f);
  }
}

void addmul3_hmatrix(real alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y, TruncationOperator &trunc, bool ztrans, phmatrix z)
{
  SumExpression S = ztrans ? SumExpression(alpha, !ytrans, y, !xtrans, x) : SumExpression(alpha, xtrans, x, ytrans, y);

  addmul_hmatrix_sumexpression(S, trunc, z);
}

/* ***********************************
 *   Forwardsolve with H-matrices    *
 *************************************/
static void lowersolve_nn_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->rc == x->rc);

  if (x->f)
  {
    s.add_amatrix(x->f);

    triangularinvmul_hmatrix_amatrix(true, aunit, false, a, false, x->f);
  }
  else if (x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);

    triangularinvmul_hmatrix_amatrix(true, aunit, false, a, false, &x->r->A);
  }
  else
  {
    for (uint ell = 0; ell < x->csons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> row_splits;
      for (uint i = 0; i < a->rsons; i++)
        row_splits.push_back(s.restrict_to(i, ell));

      for (uint i = 0; i < a->rsons; i++)
      {
        lowersolve_nn_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc,
                                            x->son[i + ell * x->rsons], *row_splits[i]);

        for (uint j = i + 1; j < a->rsons; j++)
        {
          row_splits[j]->add_product(-1.0, false, a->son[j + i * a->rsons], false, x->son[i + ell * x->rsons]);
        }
      }
    }
  }
}

static void lowersolve3_nn_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);

  lowersolve_nn_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void lowersolve_nt_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{

  assert(a->rc == a->cc);
  assert(a->rc == x->cc);

  if (x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(true, aunit, false, a, true, x->f);
  }
  else if (x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(true, aunit, false, a, false, &x->r->B);
  }
  else
  {
    for (uint ell = 0; ell < x->rsons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> col_splits;
      for (uint i = 0; i < a->rsons; i++)
        col_splits.push_back(s.restrict_to(ell, i));

      for (uint i = 0; i < a->rsons; i++)
      {
        lowersolve_nt_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[ell + i * x->rsons], *col_splits[i]);

        for (uint j = i + 1; j < a->rsons; j++)
        {
          col_splits[j]->add_product(-1.0, false, x->son[ell + i * x->rsons], true, a->son[j + i * a->rsons]);
        }
      }
    }
  }
}

static void
lowersolve3_nt_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);

  lowersolve_nt_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void lowersolve_tn_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->cc == x->rc);

  if (x->f)
  {
    s.add_amatrix(x->f);

    triangularinvmul_hmatrix_amatrix(true, aunit, true, a, false, x->f);
  }
  else if (x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);

    triangularinvmul_hmatrix_amatrix(true, aunit, true, a, false, &x->r->A);
  }
  else
  {
    for (uint ell = 0; ell < x->csons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> row_splits;
      for (uint i = 0; i < x->rsons; i++)
        row_splits.push_back(s.restrict_to(i, ell));

      for (uint i = a->rsons; i-- > 0;)
      {
        lowersolve_tn_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[i + ell * x->rsons], *row_splits[i]);

        for (uint j = i; j-- > 0;)
          row_splits[j]->add_product(-1.0, true, a->son[i + j * a->rsons], false, x->son[i + ell * x->rsons]);
      }
    }
  }
}

static void
lowersolve3_tn_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);

  lowersolve_tn_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
lowersolve_tt_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->cc == x->cc);

  if (x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(true, aunit, true, a, true, x->f);
  }
  else if (x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(true, aunit, true, a, false, &x->r->B);
  }
  else
  {
    for (uint ell = 0; ell < x->rsons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> col_splits;
      for(uint i = 0; i < x->csons; i++)
        col_splits.push_back(s.restrict_to(ell, i));

      for(uint i = a->rsons; i-- > 0;)
      {
        lowersolve_tt_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[ell + i * x->rsons], *col_splits[i]);

        for(uint j = i; j-- > 0;)
          col_splits[j]->add_product(-1.0, false, x->son[ell + i * x->rsons], false, a->son[i + j * a->rsons]);
      }
    }
  }
}

static void
lowersolve3_tt_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  lowersolve_tt_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
lowersolve_hmatrix_sumexpression(bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x, SumExpression &s)
{
  if(atrans)
  {
    if(xtrans)
      lowersolve_tt_hmatrix_sumexpression(aunit, a, trunc, x, s);
    else
      lowersolve_tn_hmatrix_sumexpression(aunit, a, trunc, x, s);
  }
  else
  {
    if(xtrans)
      lowersolve_nt_hmatrix_sumexpression(aunit, a, trunc, x, s);
    else
      lowersolve_nn_hmatrix_sumexpression(aunit, a, trunc, x, s);
  }
}

void lowersolve3_hmatrix(bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  lowersolve_hmatrix_sumexpression(aunit, atrans, a, trunc, xtrans, x, s);
}

/* ***********************************
 *   Backwardssolve with H-matrices  *
 *************************************/
static void
uppersolve_nn_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->cc == x->rc);

  if(x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(false, aunit, false, a, false, x->f);
  }
  else if(x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(false, aunit, false, a , false, &x->r->A);
  }
  else
  {
    for(uint ell = 0; ell < x->csons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> row_splits;
      for(uint i = 0; i < a->rsons; i++)
        row_splits.push_back(s.restrict_to(i, ell));

      for(uint i = a->rsons; i-- > 0;)
      {
        uppersolve_nn_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[i + ell * x->rsons], *row_splits[i]);

        for(uint j = i; j-- > 0;)
          row_splits[j]->add_product(-1.0, false, a->son[j + i * a->rsons], false, x->son[i + ell * x->rsons]);
      }
    }
  }
}

static void
uppersolve3_nn_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  uppersolve_nn_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
uppersolve_nt_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->cc == x->cc);

  if(x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(false, aunit, false, a, true, x->f);
  }
  else if(x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(false, aunit, false, a , false, &x->r->B);
  }
  else
  {
    for(uint ell = 0; ell < x->rsons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> col_splits;
      for(uint i = 0; i < a->rsons; i++)
        col_splits.push_back(s.restrict_to(ell, i));

      for(uint i = a->rsons; i-- > 0;)
      {
        uppersolve_nt_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[ell + i * x->rsons], *col_splits[i]);

        for(uint j = i; j-- > 0;)
          col_splits[j]->add_product(-1.0, false, x->son[ell + i * x->rsons], true, a->son[j + i * a->rsons]);
      }
    }
  }
}

static void
uppersolve3_nt_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  uppersolve_nt_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
uppersolve_tn_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  if(x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(false, aunit, true, a, false, x->f);
  }
  else if(x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(false, aunit, true, a, false, &x->r->A);
  }
  else
  {
    for(uint ell = 0; ell < x->csons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> row_splits;
      for(uint i = 0; i < a->rsons; i++)
        row_splits.push_back(s.restrict_to(i, ell));
      
      for(uint i = 0; i < a->rsons; i++)
      {
        uppersolve_tn_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[i + ell * x->rsons], *row_splits[i]);

        for(uint j = i+1; j < a->rsons; j++)
          row_splits[j]->add_product(-1.0, true, a->son[i + j * a->rsons], false, x->son[i + ell * x->rsons]);
      }
    }
  }
}

static void
uppersolve3_tn_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  uppersolve_tn_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
uppersolve_tt_hmatrix_sumexpression(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x, SumExpression &s)
{
  if(x->f)
  {
    s.add_amatrix(x->f);
    triangularinvmul_hmatrix_amatrix(false, aunit, true, a, true, x->f);
  }
  else if(x->r)
  {
    trunc.addtrunc_sumexpression_rkmatrix(s, x->r);
    triangularinvmul_hmatrix_amatrix(false, aunit, true, a, false, &x->r->B);
  }
  else
  {
    for(uint ell = 0; ell < x->rsons; ell++)
    {
      std::vector<std::unique_ptr<SumExpression>> col_splits;
      for(uint i = 0; i < a->rsons; i++)
        col_splits.push_back(s.restrict_to(ell, i));
      
      for(uint i = 0; i < a->rsons; i++)
      {
        uppersolve_tt_hmatrix_sumexpression(aunit, a->son[i + i * a->rsons], trunc, x->son[ell + i * x->rsons], *col_splits[i]);

        for(uint j = i+1; j < a->rsons; j++)
          col_splits[j]->add_product(-1.0, false, x->son[ell + i* x->rsons], false, a->son[i + j * a->rsons]);
      }
    }
  }
}

static void
uppersolve3_tt_hmatrix(bool aunit, pchmatrix a, TruncationOperator &trunc, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  uppersolve_tt_hmatrix_sumexpression(aunit, a, trunc, x, s);
}

static void
uppersolve_hmatrix_sumexpression(bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x, SumExpression &s)
{
  if(atrans)
  {
    if(xtrans)
      uppersolve_tt_hmatrix_sumexpression(aunit, a, trunc, x, s);
    else
      uppersolve_tn_hmatrix_sumexpression(aunit, a, trunc, x, s);
  }
  else
  {
    if(xtrans)
      uppersolve_nt_hmatrix_sumexpression(aunit, a, trunc, x, s);
    else
      uppersolve_nn_hmatrix_sumexpression(aunit, a, trunc, x, s);
  }
}


void triangularinvmul_hmatrix_sumexpression(bool alower, bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x, SumExpression &s)
{
  if (alower)
    lowersolve_hmatrix_sumexpression(aunit, atrans, a, trunc, xtrans, x, s);
  else
    uppersolve_hmatrix_sumexpression(aunit, atrans, a, trunc, xtrans, x, s);
}

void triangularinvmul3_hmatrix(bool alower, bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x)
{
  SumExpression s(x->rc, x->cc);
  triangularinvmul_hmatrix_sumexpression(alower, aunit, atrans, a, trunc, xtrans, x, s);
}

/* *********************************************
 * Compute the LU-decomposition of a H-matrix  *
 ***********************************************/
static void
lrdecomp_hmatrix_sumexpression(phmatrix a, TruncationOperator &trunc, SumExpression &s)
{
  assert(a->rc == a->cc);
  assert(a->r == nullptr);

  if(a->f)
  {
    s.add_amatrix(a->f);
    lrdecomp_amatrix(a->f);
  }
  else
  {
    std::vector<std::unique_ptr<SumExpression>> split_s(a->rsons * a->csons);
    for(uint j = 0; j < a->csons; j++)
      for(uint i = 0; i < a->rsons; i++)
        split_s[i+j*a->rsons] = s.restrict_to(i, j);

    for(uint ell = 0; ell < a->rsons; ell++)
    {
      lrdecomp_hmatrix_sumexpression(a->son[ell + ell * a->rsons], trunc, *split_s[ell + ell * a->rsons]);

      // Solve for upper and lower triangular part
      for(uint i = ell + 1; i < a->csons; i++)
      {
        lowersolve_hmatrix_sumexpression(true, false, a->son[ell + ell * a->rsons], trunc, false, a->son[ell + i * a->rsons], *split_s[ell + i * a->rsons]);
        uppersolve_hmatrix_sumexpression(false, true, a->son[ell + ell * a->rsons], trunc, true, a->son[i + ell * a->rsons], *split_s[i + ell * a->rsons]);
      }

      for(uint i = ell + 1; i < a->rsons; i++)
      {
        for(uint j = ell + 1; j < a->csons; j++)
        {
          split_s[i + j * a->rsons]->add_product(-1.0, false, a->son[i + ell * a->rsons], false, a->son[ell + j * a->rsons]);
        }
      }
    }
  }
}

void lrdecomp3_hmatrix(phmatrix a, TruncationOperator &trunc)
{
  SumExpression s(a->rc, a->cc);
  lrdecomp_hmatrix_sumexpression(a, trunc, s);
}
