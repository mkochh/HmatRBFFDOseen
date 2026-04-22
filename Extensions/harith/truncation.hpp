/**
 * \file truncation.h
 * \author Jonas Grams (jonas.grams@tuhh.de)
 * \brief Module containing several truncation operators for low-rank matrices.
 * \version 0.1
 *
 */



#ifndef TRUNC_HEADER
#define TRUNC_HEADER

#include "../../H2Lib/h2lib.h"
#include "sumexpression.hpp"
#include <memory>

class TruncationOperator
{
public:
  TruncationOperator(){};

  virtual void truncate_rkmatrix(prkmatrix){};

  virtual void addtrunc_rkmatrix(field, pcrkmatrix, prkmatrix){};

  virtual void addmultrunc_hmatrix_rkmatrix(field, bool, pchmatrix,
                                            bool, pchmatrix, prkmatrix){};

  virtual void addtrunc_sumexpression_rkmatrix(SumExpression &, prkmatrix){};
};

class SVDTruncation : public TruncationOperator
{
public:
  real eps;
  ptruncmode tm;

  SVDTruncation(real eps, ptruncmode tm = nullptr): eps(eps), tm(tm)
  {
    if (this->tm == nullptr)
    {
      this->tm = new_releucl_truncmode();
    }
  }

  ~SVDTruncation()
  {
    del_truncmode(this->tm);
  };

  void truncate_rkmatrix(prkmatrix rk) override;
  void addtrunc_rkmatrix(field alpha, pcrkmatrix x, prkmatrix y) override;
  void addmultrunc_hmatrix_rkmatrix(field alpha, bool xtrans, pchmatrix x,
                                    bool ytrans, pchmatrix y, prkmatrix z) override;
  void addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix z) override;
};
#endif