#include <iostream>
#include <vector>
#include "rand_trunc.hpp"

#ifdef USE_FLOAT
extern "C" void slatsqr_(unsigned int *m, unsigned int *n, unsigned int *mb,
                         unsigned int *nb, float *a, unsigned int *lda, float *tau, unsigned int *ldt,
                         float *work, int *lwork, int *info);

extern "C" void sorgtsqr_(unsigned int *m, unsigned int *n, unsigned int *mb,
                          unsigned int *nb, flaot *a, unsigned int *lda, float *tau, unsigned int *ldt,
                          float *work, int *lwork, int *info);
#else
extern "C" void dlatsqr_(unsigned int *m, unsigned int *n, unsigned int *mb,
                         unsigned int *nb, double *a, unsigned int *lda, double *tau, unsigned int *ldt,
                         double *work, int *lwork, int *info);

extern "C" void dorgtsqr_(unsigned int *m, unsigned int *n, unsigned int *mb,
                          unsigned int *nb, double *a, unsigned int *lda, double *tau, unsigned int *ldt,
                          double *work, int *lwork, int *info);
#endif

static void thin_qrdecomp_orthogonal_factor(pamatrix a, uint nb = 1)
{
    assert(a->rows >= a->cols);

    int lwork = nb * a->cols;
    field *work = allocfield(lwork);

    pamatrix tau = new_amatrix(nb, a->cols);

    int info;
    dlatsqr_(&a->rows, &a->cols, &a->rows, &nb, a->a, &a->ld, tau->a, &tau->ld, work, &lwork, &info);
    assert(info == 0);

    freemem(work);

    lwork = (a->rows + nb) * a->cols;
    work = allocfield(lwork);

    dorgtsqr_(&a->rows, &a->cols, &a->rows, &nb, a->a, &a->ld, tau->a, &tau->ld, work, &lwork, &info);
    assert(info == 0);

    freemem(work);
    del_amatrix(tau);
}

static void
randn_avector(pavector x)
{
    uint i;
    real x1, x2;

    for (i = 0; i < x->dim; i++)
    {
        x1 = 1.0 * rand() / RAND_MAX;
        x2 = 1.0 * rand() / RAND_MAX;
        x->v[i] = cos(2 * M_PI * x1) * sqrt(-2.0 * log(x2));
    }
}

void rand_addtrunc_matrix(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint rank, uint orthosteps, real tol)
{
    uint rows = trg->A.rows;
    uint cols = trg->B.rows;
    uint ell = UINT_MIN3(rows, cols, rank);
    std::vector<pavector> L(0);
    std::vector<pavector> R(0);
    // pamatrix L = new_zero_amatrix(rows, ell);
    // pamatrix R = new_zero_amatrix(cols, ell);

    real normfrob = 0.0;
    uint k = 0;
    while (k < ell)
    {
        pavector w = new_avector(cols);
        randn_avector(w);
        pavector l = new_zero_avector(rows);
        mvm_matrix(alpha, false, M, w, l);
        mvm_rkmatrix_avector(1.0, false, trg, w, l);

        del_avector(w);
        for (uint i = 0; i < orthosteps; i++)
        {
            for (pavector lk : L)
            {
                real gamma = dotprod_avector(lk, l);
                add_avector(-gamma, lk, l);
            }
        }

        real norm = norm2_avector(l);

        // Quick exit if ||l|| = 0 up to machine accuracy
        if (norm < 1e-15)
        {
            del_avector(l); // Delete l since it is not added to L.
            break;
        }
        scale_avector(1.0 / norm, l);
        L.push_back(l);

        pavector r = new_zero_avector(cols);
        mvm_matrix(alpha, true, M, l, r);
        mvm_rkmatrix_avector(1.0, true, trg, l, r);
        real eta = norm2_avector(r);

        R.push_back(r);
        
        if (eta <= tol * REAL_SQRT(normfrob))
            break;

        normfrob += eta * eta;
        k += 1;
    }

    resize_rkmatrix(trg, rows, cols, k);
    for (uint i = 0; i < k; i++)
    {
        avector ai;
        avector bi;

        init_column_avector(&ai, &trg->A, i);
        init_column_avector(&bi, &trg->B, i);

        copy_avector(L[i], &ai);
        copy_avector(R[i], &bi);

        uninit_avector(&ai);
        uninit_avector(&bi);
    }
    for (pavector l : L)
    {
        del_avector(l);
        (void)l;
    }
    for (pavector r : R)
    {
        del_avector(r);
        (void)r;
    }
}

void rand_addtrunc_matrix_debug(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint rank, uint orthosteps, real tol, uint q, std::vector<double> &err)
{
    uint rows = trg->A.rows;
    uint cols = trg->B.rows;
    uint ell = UINT_MIN3(rows, cols, rank);
    pamatrix L = new_zero_amatrix(rows, ell);
    pamatrix R = new_zero_amatrix(cols, ell);

    real normfrob = 0.0;
    uint k = 0;
    while (k < ell)
    {
        pavector w = new_avector(cols);
        randn_avector(w);

        avector tmpl;
        pavector l = init_column_avector(&tmpl, L, k);
        mvm_matrix(alpha, false, M, w, l);
        mvm_rkmatrix_avector(1.0, false, trg, w, l);

        del_avector(w);

        pamatrix Lk = new_sub_amatrix(L, rows, 0, k, 0);
        for (uint i = 0; i < orthosteps; i++)
        {
            pavector tmp = new_zero_avector(k);
            addevaltrans_amatrix_avector(1.0, Lk, l, tmp);
            addeval_amatrix_avector(-1.0, Lk, tmp, l);
            del_avector(tmp);
        }
        del_amatrix(Lk);

        real norm = norm2_avector(l);

        if (norm == 0.0)
            break;

        scale_avector(1.0 / norm, l);
        uninit_avector(l);

        // Do q subspace iterations
        for (uint a = 0; a < q; a++)
        {
            avector tmpr;
            pavector r = init_column_avector(&tmpr, R, k);
            avector tmpl;
            pavector l = init_column_avector(&tmpl, L, k);

            clear_avector(r);
            mvm_matrix(alpha, true, M, l, r);
            mvm_rkmatrix_avector(1.0, true, trg, l, r);

            pamatrix Rk = new_sub_amatrix(R, cols, 0, k, 0);
            for (uint i = 0; i < orthosteps; i++)
            {
                pavector tmp = new_zero_avector(k);
                addevaltrans_amatrix_avector(1.0, Rk, r, tmp);
                addeval_amatrix_avector(-1.0, Rk, tmp, r);
                del_avector(tmp);
            }
            del_amatrix(Rk);

            norm = norm2_avector(r);
            scale_avector(1.0 / norm, r);

            clear_avector(l);
            mvm_matrix(alpha, false, M, r, l);
            mvm_rkmatrix_avector(1.0, false, trg, r, l);

            pamatrix Lk = new_sub_amatrix(L, rows, 0, k, 0);
            for (uint i = 0; i < orthosteps; i++)
            {
                pavector tmp = new_zero_avector(k);
                addevaltrans_amatrix_avector(1.0, Lk, l, tmp);
                addeval_amatrix_avector(-1.0, Lk, tmp, l);
                del_avector(tmp);
            }
            del_amatrix(Lk);

            norm = norm2_avector(l);
            scale_avector(1.0 / norm, l);

            uninit_avector(l);
            uninit_avector(r);
        }

        avector tmpr;
        pavector r = init_column_avector(&tmpr, R, k);
        l = init_column_avector(&tmpl, L, k);
        clear_avector(r);
        mvm_matrix(alpha, true, M, l, r);
        mvm_rkmatrix_avector(1.0, true, trg, l, r);
        real eta = norm2_avector(r);
        err.push_back(eta);

        uninit_avector(l);
        uninit_avector(r);

        if (eta <= tol * REAL_SQRT(normfrob))
            break;

        normfrob += eta * eta;
        k += 1;
    }

    resize_rkmatrix(trg, rows, cols, k);
    copy_sub_amatrix(false, L, &trg->A);
    copy_sub_amatrix(false, R, &trg->B);

    del_amatrix(L);
    del_amatrix(R);
}

void RandomTruncation::addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix trg)
{
    // do not truncate if it is only one rk-matrix (the matrix itself)
    if (src.h_sum_list.size() > 0 || src.rk_sum_list.size() > 1) {
        uint r = UINT_MIN(src.rc->size, src.cc->size);
        rand_addtrunc_matrix(1.0, &src, (mvm_t)SumExpression::mvm, trg, r, this->orthosteps, this->eps);
    }
}
