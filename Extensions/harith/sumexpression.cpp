#include <assert.h>
#include "sumexpression.hpp"
#include "harith3.hpp"

static inline phmatrix getson_hmatrix(uint ridx, uint cidx, pchmatrix trg)
{
    return trg->son[ridx + cidx * trg->rsons];
}

static void
addmul_hmatrix_amatrix(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y, pamatrix trg)
{
    assert((xtrans ? x->cc->size : x->rc->size) == trg->rows);
    assert((ytrans ? y->rc->size : y->cc->size) == trg->cols);

    assert((xtrans ? x->rc : x->cc) == (ytrans ? y->cc : y->rc));

    if (x->r)
    {
        pamatrix xA = xtrans ? getB_rkmatrix(x->r) : getA_rkmatrix(x->r);
        pamatrix xB = xtrans ? getA_rkmatrix(x->r) : getB_rkmatrix(x->r);

        pamatrix tmp = new_zero_amatrix(trg->cols, x->r->k);
        // tmp <- y(T)^T xB
        addmul_hmatrix_amatrix_amatrix(1.0, !ytrans, y, false, xB, false, tmp);
        // trg <- trg + xA tmp^T
        addmul_amatrix(alpha, false, xA, true, tmp, trg);
        del_amatrix(tmp);
    }
    else if (x->f)
    {
        // trg <- trg + x(T) y(T) = (trg^T + y(T)^T x(T)^T)^T
        addmul_hmatrix_amatrix_amatrix(alpha, !ytrans, y, !xtrans, x->f, true, trg);
    }
    else if (y->r)
    {
        pamatrix yA = ytrans ? getB_rkmatrix(y->r) : getA_rkmatrix(y->r);
        pamatrix yB = ytrans ? getA_rkmatrix(y->r) : getB_rkmatrix(y->r);

        pamatrix tmp = new_zero_amatrix(trg->rows, y->r->k);
        // tmp <- x(T) yA
        addmul_hmatrix_amatrix_amatrix(1.0, xtrans, x, false, yA, false, tmp);
        // trg <- trg + tmp yB^T
        addmul_amatrix(alpha, false, tmp, true, yB, trg);
        del_amatrix(tmp);
    }
    else if (y->f)
    {
        // trg <- trg + x(T) y(T)
        addmul_hmatrix_amatrix_amatrix(alpha, xtrans, x, ytrans, y->f, false, trg);
    }
    else
    {
        assert(x->son != nullptr);
        assert(y->son != nullptr);

        pccluster rc = xtrans ? x->cc : x->rc;
        pccluster cc = ytrans ? y->rc : y->cc;
        bool rsplit = xtrans ? (x->cc != x->son[0]->cc) : (x->rc != x->son[0]->rc);
        bool csplit = ytrans ? (y->rc != y->son[0]->rc) : (y->cc != y->son[0]->cc);

        phmatrix split = split_sub_amatrix(trg, rc, cc, rsplit, csplit);
        TruncationOperator trunc; // No truncation necessary
        addmul3_hmatrix(alpha, xtrans, x, ytrans, y, trunc, false, split);
        del_hmatrix(split);
    }
}

void HSummand::addmul_hsummand_amatrix(pamatrix trg)
{
    assert((xtrans ? x->cc->size : x->rc->size) == trg->rows);
    assert((ytrans ? y->rc->size : y->cc->size) == trg->cols);

    assert((xtrans ? x->rc : x->cc) == (ytrans ? y->cc : y->rc));

    if (x->f)
    {
        // trg^T <- trg^T + y(T)^T x(T)^T
        addmul_hmatrix_amatrix_amatrix(alpha, !ytrans, y, !xtrans, x->f, true, trg);
    }
    else if (y->f)
    {
        addmul_hmatrix_amatrix_amatrix(alpha, xtrans, x, ytrans, y->f, false, trg);
    }
    else
    {
        assert(x->son != nullptr);
        assert(y->son != nullptr);

        pccluster rc = xtrans ? x->cc : x->rc;
        pccluster cc = ytrans ? y->rc : y->cc;
        bool rsplit = xtrans ? (x->cc != x->son[0]->cc) : (x->rc != x->son[0]->rc);
        bool csplit = ytrans ? (y->rc != y->son[0]->rc) : (y->cc != y->son[0]->cc);

        phmatrix split = split_sub_amatrix(trg, rc, cc, rsplit, csplit);
        TruncationOperator trunc; // No truncation necessary
        addmul3_hmatrix(alpha, xtrans, x, ytrans, y, trunc, false, split);
        del_hmatrix(split);
    }
}

static void
addeval_hsummand_avector(field alpha, const HSummand *entry, pcavector x, pavector y)
{
    pccluster xmid, ymid;
    xmid = entry->xtrans ? entry->x->rc : entry->x->cc;
    ymid = entry->ytrans ? entry->y->cc : entry->y->rc;

    assert(xmid == ymid);

    pavector tmp;
    tmp = new_zero_avector(xmid->size);

    // Apply entry->y to x
    if (entry->ytrans)
        fastaddevaltrans_hmatrix_avector(1.0, entry->y, x, tmp);
    else
        fastaddeval_hmatrix_avector(1.0, entry->y, x, tmp);

    // Apply entry->x to tmp
    if (entry->xtrans)
        fastaddevaltrans_hmatrix_avector(alpha*entry->alpha, entry->x, tmp, y);
    else
        fastaddeval_hmatrix_avector(alpha*entry->alpha, entry->x, tmp, y);

    // Clean up
    del_avector(tmp);
}

static void
addevaltrans_hsummand_avector(field alpha, const HSummand *entry, pcavector x, pavector y)
{
    pccluster xmid, ymid;
    xmid = entry->xtrans ? entry->x->rc : entry->x->cc;
    ymid = entry->ytrans ? entry->y->cc : entry->y->rc;

    assert(xmid == ymid);

    pavector tmp;
    tmp = new_zero_avector(xmid->size);

    // Apply entry->y to x
    if (entry->xtrans)
        fastaddeval_hmatrix_avector(1.0, entry->x, x, tmp);
    else
        fastaddevaltrans_hmatrix_avector(1.0, entry->x, x, tmp);

    // Apply entry->x to tmp
    if (entry->ytrans)
        fastaddeval_hmatrix_avector(alpha*entry->alpha, entry->y, tmp, y);
    else
        fastaddevaltrans_hmatrix_avector(alpha*entry->alpha, entry->y, tmp, y);

    // Clean up
    del_avector(tmp);
}

void HSummand::mvm(field alpha, bool atrans, HSummand *a, pcavector x, pavector y)
{
    if (atrans)
    {
        addevaltrans_hsummand_avector(alpha, a, x, y);
    }
    else
    {
        addeval_hsummand_avector(alpha, a, x, y);
    }
}

SumExpression::SumExpression(pccluster rc, pccluster cc) : rk_sum_list(), h_sum_list()
{
    this->rc = rc;
    this->cc = cc;
}

SumExpression::SumExpression(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y) : rk_sum_list(), h_sum_list()
{
    this->rc = xtrans ? x->cc : x->rc;
    this->cc = ytrans ? y->rc : y->cc;

    this->add_product(alpha, xtrans, x, ytrans, y);
}

void SumExpression::add_product(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y)
{
    pccluster rc = xtrans ? x->cc : x->rc;
    pccluster cc = ytrans ? y->rc : y->cc;

    assert((xtrans ? x->rc : x->cc) == (ytrans ? y->cc : y->rc));

    uint rank_x = -1, rank_y = -1;
    if (x->r != nullptr)
        rank_x = getrank_rkmatrix(x->r);
    if (y->r != nullptr)
        rank_y = getrank_rkmatrix(y->r);

    // do nothing if either of the matrices x and y has rank zero
    if (rank_x != 0 && rank_y != 0) {
        // If either x or y is an rk-matrix, add the product to the low-rank sum list
        if (x->r != nullptr)
        {
            uint k = rank_x;
            prkmatrix rk = new_rkmatrix(rc->size, cc->size, k);

            pamatrix xA = xtrans ? getB_rkmatrix(x->r) : getA_rkmatrix(x->r);
            pamatrix xB = xtrans ? getA_rkmatrix(x->r) : getB_rkmatrix(x->r);

            // rk->A <- xA
            copy_amatrix(false, xA, &rk->A);
            // rk->B <- y(T)^T  xB
            clear_amatrix(&rk->B);
            addmul_hmatrix_amatrix_amatrix(alpha, !ytrans, y, false, xB, false, &rk->B);

            this->rk_sum_list.push_back(rk);
        }
        else if (y->r != nullptr)
        {
            uint k = rank_y;
            prkmatrix rk = new_rkmatrix(rc->size, cc->size, k);

            // y(T) = yA yB^{T}
            pamatrix yA = ytrans ? getB_rkmatrix(y->r) : getA_rkmatrix(y->r);
            pamatrix yB = ytrans ? getA_rkmatrix(y->r) : getB_rkmatrix(y->r);

            // rk->B <- yB
            copy_amatrix(false, yB, &rk->B);
            // rk->A <- x(T) yA
            clear_amatrix(&rk->A);
            addmul_hmatrix_amatrix_amatrix(alpha, xtrans, x, false, yA, false, &rk->A);

            this->rk_sum_list.push_back(rk);
        }
        else // If x and y are both either dense or h-matrices, add them to the list of h-matrix summands
        {
            this->h_sum_list.emplace_back(alpha, xtrans, x, ytrans, y);
        }
    }
}

std::unique_ptr<SumExpression> SumExpression::restrict_to(uint rson, uint cson)
{
    pccluster rcson = (rc->sons > 0) ? rc->son[rson] : rc;
    pccluster ccson = (cc->sons > 0) ? cc->son[cson] : cc;

    std::unique_ptr<SumExpression> restriction(new SumExpression(rcson, ccson));

    for (auto &entry : h_sum_list)
    {
        // If x is dense, split x
        phmatrix xsplit = nullptr;
        phmatrix ysplit = nullptr;

        if (entry.x->f)
        {
            // Split the columns of x/x^T only, if the columns of y are split as well.
            bool xmidsplit;
            if (entry.y->f)
                xmidsplit = false;
            else
                xmidsplit = entry.ytrans ? (entry.y->cc != entry.y->son[0]->cc) : (entry.y->rc != entry.y->son[0]->rc);

            bool xrsplit = entry.xtrans ? xmidsplit : true;
            bool xcsplit = entry.xtrans ? true : xmidsplit;

            xsplit = split_sub_amatrix(entry.x->f, entry.x->rc, entry.x->cc, xrsplit, xcsplit);

            this->split_matrices.push_back(xsplit);
        }
        if (entry.y->f)
        {
            // Split the rows of y/y^T only, if the columns of x are split as well.
            bool ymidsplit;
            if (entry.x->f)
                ymidsplit = false;
            else
                ymidsplit = entry.xtrans ? (entry.x->rc != entry.x->son[0]->rc) : (entry.x->cc != entry.x->son[0]->cc);

            bool yrsplit = entry.ytrans ? true : ymidsplit;
            bool ycsplit = entry.ytrans ? ymidsplit : true;

            ysplit = split_sub_amatrix(entry.y->f, entry.y->rc, entry.y->cc, yrsplit, ycsplit);

            this->split_matrices.push_back(ysplit);
        }

        uint msons;
        if (xsplit != nullptr)
            msons = (entry.xtrans ? xsplit->rsons : xsplit->csons);
        else
            msons = (entry.xtrans ? entry.x->rsons : entry.x->csons);
        // Add sub-products with rc->son[i] as row-cluster and cc->son[i] as column cluster
        for (uint k = 0; k < msons; k++)
        {
            pchmatrix xson;
            if (xsplit != nullptr) // entry.x was a dense matrix and is split
                xson = entry.xtrans ? getson_hmatrix(k, rson, xsplit) : getson_hmatrix(rson, k, xsplit);
            else // entry.x has sub-matrices
                xson = entry.xtrans ? getson_hmatrix(k, rson, entry.x) : getson_hmatrix(rson, k, entry.x);

            pchmatrix yson;
            if (ysplit != nullptr) // entry.y was a dense matrix and is split
                yson = entry.ytrans ? getson_hmatrix(cson, k, ysplit) : getson_hmatrix(k, cson, ysplit);
            else
                yson = entry.ytrans ? getson_hmatrix(cson, k, entry.y) : getson_hmatrix(k, cson, entry.y);

            restriction->add_product(entry.alpha, entry.xtrans, xson, entry.ytrans, yson);
        }
    }

    uint roff = 0;
    for (uint i = 0; i < rson; i++)
    {
        roff += rc->son[i]->size;
    }
    uint coff = 0;
    for (uint j = 0; j < cson; j++)
    {
        coff += cc->son[j]->size;
    }

    for (prkmatrix entry : rk_sum_list)
    {
        prkmatrix rk = (prkmatrix)new_sub_rkmatrix(entry, rcson->size, roff, ccson->size, coff);

        restriction->rk_sum_list.push_back(rk);
    }

    return restriction;
}

void SumExpression::add_amatrix(pamatrix a)
{
    // Add H-Matrix sum list
    for (HSummand &entry : h_sum_list)
    {
        entry.addmul_hsummand_amatrix(a);
    }

    // Add rkmatrix sum list
    for (prkmatrix entry : rk_sum_list)
    {
        addmul_amatrix(1.0, false, &entry->A, true, &entry->B, a);
    }
}

void SumExpression::mvm(field alpha, bool atrans, SumExpression *a, pcavector x, pavector y)
{
    for (HSummand &entry : a->h_sum_list)
    {
        HSummand::mvm(alpha, atrans, &entry, x, y);
    }

    if (atrans)
    {
        for (prkmatrix entry : a->rk_sum_list)
        {
            addevaltrans_rkmatrix_avector(alpha, entry, x, y);
        }

    }
    else 
    {
        for (prkmatrix entry : a->rk_sum_list)
        {
            addeval_rkmatrix_avector(alpha, entry, x, y);
        }

    }
}