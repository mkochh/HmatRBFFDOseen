
#ifndef SUMEXPRESSION_HEADER
#define SUMEXPRESSION_HEADER

#include <iostream>
#include <list>
#include <memory>
#include "../../H2Lib/h2lib.h"
class HSummand
{
public:
    field alpha;

    bool xtrans;
    pchmatrix x;
    // phmatrix xsplit; // Auxilliary pointer used if x is dense and has to be split.

    bool ytrans;
    pchmatrix y;
    // phmatrix ysplit;// Auxilliary pointer used if y is dense and has to be split.

    HSummand(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y) //, phmatrix xsplit = nullptr, 
        // phmatrix ysplit = nullptr)
    {
        this->alpha = alpha;

        this->xtrans = xtrans;
        this->x = x;

        this->ytrans = ytrans;
        this->y = y;
    };

    void addmul_hsummand_amatrix(pamatrix trg);

    static void mvm(field alpha, bool atrans, HSummand *a, pcavector x, pavector y);
};
class SumExpression
{
public:
    /**
     * @brief Row cluster of the sum expression.
     *
     */
    pccluster rc;
    /**
     * @brief Column cluster of the sum expression.
     *
     */
    pccluster cc;
    /**
     * @brief List of rkmatrix summands.
     *
     */
    std::list<prkmatrix> rk_sum_list;
    /**
     * @brief List of H-matrix summands.
     *
     */
    std::list<HSummand> h_sum_list;

    /**
     * @brief List of hmatrices generated to split dense matrices.
     *  They are collected in this list and deleted with the SumExpression
     *  object.
    */
    std::list<phmatrix> split_matrices;

    /**
     * @brief Construct a new Sum Expression object. The row and column clusters
     *      are given as arguments, while rk_sum_list and h_sum_list are initialized
     *      as empty vectors.
     *
     * @param rc
     * @param cc
     */
    SumExpression(pccluster rc, pccluster cc);

    ~SumExpression()
    {
        for(prkmatrix r : rk_sum_list)
        {
            del_rkmatrix(r);
        }

        for(phmatrix entry : split_matrices)
        {
            del_hmatrix(entry);
        }
    };

    /**
     * @brief Construct a new Sum Expression object with a given H-Matrix product.
     *  The row cluster is determined from the left factor x and the column cluster
     *  is determined from the right factor y. Initializes either rk_sum_list or h_sum_list
     *  with one element.
     *
     * @param xtrans Transpose flag for the left factor x.
     * @param x Left factor of the H-Matrix product.
     * @param ytrans Transpose flag for the right factor y.
     * @param y Right factor of the H-Matrix product.
     */
    SumExpression(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y);

    void add_product(field alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y);

    void add_amatrix(pamatrix a);

    /**
     * @brief Create sum expressions restricted to the given row and column son cluster.
     *
     */
    std::unique_ptr<SumExpression> restrict_to(uint rson, uint cson);

    static void mvm(field alpha, bool atrans, SumExpression *a, pcavector x, pavector y);
};

#endif