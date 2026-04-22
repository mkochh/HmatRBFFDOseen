#ifndef RAND_TRUNC_HEADER
#define RAND_TRUNC_HEADER

#include "truncation.hpp"
#include "sumexpression.hpp"
#include <vector>

typedef void (*addmul_t)(field, bool, void *, bool, pcamatrix, pamatrix);

prkmatrix
randtrunc_matrix_lvl1(void *M, mvm_t mvm_matrix, uint rows, uint cols, uint r, uint orthosteps, real tol);

prkmatrix
randtrunc_matrix_lvl2(void *M, addmul_t addmul_matrix, uint rows, uint cols, uint ell,
                      uint orthosteps, real tol, uint block_size);

void rand_addtrunc_matrix(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint rank, uint orthosteps, real tol);
void rand_addtrunc_matrix_debug(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint rank, uint orthosteps, real tol, uint q, std::vector<double> &err);
class RandomTruncation: public TruncationOperator
{
public:
    real eps;
    uint orthosteps;

    RandomTruncation(real eps, uint orthosteps = 1): eps(eps), orthosteps(orthosteps){};

    void addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix trg) override;
};

#endif