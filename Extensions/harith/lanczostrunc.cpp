#include <vector>
#include "lanczostrunc.hpp"

void lanczos_addtrunc_rkmatrix(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint r, real tol, uint orthosteps)
{
    uint rows = trg->A.rows;
    uint cols = trg->B.rows;
    uint ell = UINT_MIN3(trg->A.rows, trg->B.rows, r);
    std::vector<pavector> w;
    std::vector<pavector> q;
    pavector a = new_zero_avector(ell);
    pavector b = new_zero_avector(ell);

    pavector w0 = new_avector(cols);
    random_avector(w0);
    real normw = norm2_avector(w0);
    scale_avector(1.0 / normw, w0);

    w.push_back(w0);

    uint k = 0;
    real normfrob = 0.0;
    while (k < ell)
    {
        pavector qk = new_zero_avector(rows);
        if (k > 0)
        {
            add_avector(-b->v[k - 1], q[k - 1], qk);
        }
        mvm_matrix(alpha, false, M, w[k], qk);
        mvm_rkmatrix_avector(1.0, false, trg, w[k], qk);

        // Additional reorthogonalization steps
        for (uint j = 0; j < orthosteps; j++)
        {
            for (pavector qk1 : q)
            {
                field gamma = dotprod_avector(qk, qk1);
                add_avector(-gamma, qk1, qk);
            }
        }

        a->v[k] = norm2_avector(qk);
        if(a->v[k] == 0.0)
            break;
        scale_avector(1.0 / a->v[k], qk);

        q.push_back(qk);

        real eta = k > 0 ? a->v[k] * a->v[k] + b->v[k - 1] * b->v[k - 1] : a->v[k] * a->v[k];

        if (REAL_SQRT(eta) <= tol * REAL_SQRT(normfrob))
            break;

        normfrob += eta;

        pavector wk1 = new_zero_avector(cols);
        add_avector(-a->v[k], w[k], wk1);
        mvm_matrix(alpha, true, M, qk, wk1);
        mvm_rkmatrix_avector(1.0, true, trg, qk, wk1);

        // Additional reorthogonalization steps
        for (uint j = 0; j < orthosteps; j++)
        {
            for (pavector wk : w)
            {
                field gamma = dotprod_avector(wk, wk1);
                add_avector(-gamma, wk, wk1);
            }
        }

        b->v[k] = norm2_avector(wk1);
        if (b->v[k] == 0.0)
            break;
        scale_avector(1.0 / b->v[k], wk1);

        w.push_back(wk1);

        k += 1;
    }

    resize_rkmatrix(trg, rows, cols, k);
    for (uint i = 0; i < k; i++)
    {
        avector ai, bi;
        init_column_avector(&ai, &trg->A, i);
        init_column_avector(&bi, &trg->B, i);

        copy_avector(q[i], &ai);
        copy_avector(w[i], &bi);

        uninit_avector(&ai);
        uninit_avector(&bi);
    }
    
    if(k > 0)
    {
        pavector d = new_sub_avector(a, k, 0);
        pavector l = new_sub_avector(b, k - 1, 0);
        bidiagmul_amatrix(1.0, false, &trg->B, d, l);
        del_avector(d);
        del_avector(l);
    }

    for (pavector vec : q)
    {
        if (vec != nullptr)
            del_avector(vec);
    }
    for (pavector vec : w)
    {
        if (vec != nullptr)
            del_avector(vec);
    }

    del_avector(a);
    del_avector(b);
}

void lanczos_addtrunc_rkmatrix_debug(field alpha, void *M, mvm_t mvm_matrix, prkmatrix trg, uint r, real tol, uint orthosteps, std::vector<double> &err)
{
    uint rows = trg->A.rows;
    uint cols = trg->B.rows;
    uint ell = UINT_MIN3(trg->A.rows, trg->B.rows, r);
    std::vector<pavector> w;
    std::vector<pavector> q;
    pavector a = new_zero_avector(ell);
    pavector b = new_zero_avector(ell);

    pavector w0 = new_avector(cols);
    random_avector(w0);
    real normw = norm2_avector(w0);
    scale_avector(1.0 / normw, w0);

    w.push_back(w0);

    uint k = 0;
    real normfrob = 0.0;
    while (k < ell)
    {
        pavector qk = new_zero_avector(rows);
        if (k > 0)
        {
            add_avector(-b->v[k - 1], q[k - 1], qk);
        }
        mvm_matrix(alpha, false, M, w[k], qk);
        mvm_rkmatrix_avector(1.0, false, trg, w[k], qk);

        
        
        // Additional reorthogonalization steps
        for (uint j = 0; j < orthosteps; j++)
        {
            for (pavector qk1 : q)
            {
                field gamma = dotprod_avector(qk, qk1);
                add_avector(-gamma, qk1, qk);
            }
        }
        
        real normq = norm2_avector(qk);
        a->v[k] = normq;
        if(normq == 0.0)
            break;
        
        scale_avector(1.0 / normq, qk);

        q.push_back(qk);

        real eta = k > 0 ? b->v[k-1] * b->v[k-1] : 1.0;

        err.push_back(eta / normfrob);

        if (REAL_SQRT(eta) <= tol * REAL_SQRT(normfrob))
            break;

        normfrob += eta;

        pavector wk1 = new_zero_avector(cols);
        add_avector(-a->v[k], w[k], wk1);
        mvm_matrix(alpha, true, M, qk, wk1);
        mvm_rkmatrix_avector(1.0, true, trg, qk, wk1);
        
        // Additional reorthogonalization steps
        for (uint j = 0; j < orthosteps; j++)
        {
            for (pavector wk : w)
            {
                field gamma = dotprod_avector(wk, wk1);
                add_avector(-gamma, wk, wk1);
            }
        }
        real normw = norm2_avector(wk1);
        b->v[k] = normw;
        if (normw == 0.0)
            break;
        scale_avector(1.0 / normw, wk1);

        w.push_back(wk1);

        k += 1;
    }

    resize_rkmatrix(trg, rows, cols, k);
    for (uint i = 0; i < k; i++)
    {
        avector ai, bi;
        init_column_avector(&ai, &trg->A, i);
        init_column_avector(&bi, &trg->B, i);

        copy_avector(q[i], &ai);
        copy_avector(w[i], &bi);

        uninit_avector(&ai);
        uninit_avector(&bi);
    }

    // pavector d = new_sub_avector(a, k, 0);
    // pavector l = new_sub_avector(b, k - 1, 0);
    // bidiagmul_amatrix(1.0, true, &trg->A, d, l);
    // del_avector(d);
    // del_avector(l);

    for (pavector vec : q)
    {
        if (vec != nullptr)
            del_avector(vec);
    }
    for (pavector vec : w)
    {
        if (vec != nullptr)
            del_avector(vec);
    }

    del_avector(a);
    del_avector(b);
}

void LanczosTruncation::addtrunc_sumexpression_rkmatrix(SumExpression &src, prkmatrix trg)
{
    // do not truncate if it is only one rk-matrix (the matrix itself)
    if (src.h_sum_list.size() > 0 || src.rk_sum_list.size() > 1) {
        uint r = UINT_MIN(trg->A.rows, trg->B.rows);
        lanczos_addtrunc_rkmatrix(1.0, &src, (mvm_t)SumExpression::mvm, trg, r, this->eps, this->orthosteps);
    }
}