/**
 * @file harith3.hpp
 * @author Jonas Grams (jonas.grams@tuhh.de)
 *
 * @brief Implementation of the H-Matrix multiplication
 * introduced in "On the best approximation of the hierarchical
 * matrix product" (https://arxiv.org/abs/1805.08998)
 *
 * @version 0.2
 * @date 2024-05-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef HARITH3_HEADER
#define HARITH3_HEADER

#include "../../H2Lib/h2lib.h"
#include "truncation.hpp"
#include "sumexpression.hpp"

void addmul3_hmatrix(real alpha, bool xtrans, pchmatrix x, bool ytrans, pchmatrix y, TruncationOperator &trunc, bool ztrans, phmatrix z);

void triangularinvmul3_hmatrix(bool alower, bool aunit, bool atrans, pchmatrix a, TruncationOperator &trunc, bool xtrans, phmatrix x);

void lrdecomp3_hmatrix(phmatrix a, TruncationOperator &trunc);

#endif