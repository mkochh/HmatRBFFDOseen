#include "truncation.hpp"
#include <vector>

void
lanczos_addtrunc_rkmatrix(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint r, real tol, uint orthosteps);
void lanczos_addtrunc_rkmatrix_debug(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint r, real tol, uint orthosteps, std::vector<double> &err);
class LanczosTruncation: public TruncationOperator
{
public:
    real eps;
    uint orthosteps;

    LanczosTruncation(real eps, uint orthosteps = 2): eps(eps), orthosteps(orthosteps){};

    void addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix trg) override;
};