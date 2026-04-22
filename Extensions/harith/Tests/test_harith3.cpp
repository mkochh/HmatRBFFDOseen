#include <stdio.h>
#include <cassert>
#include "../../../H2Lib/h2lib.h"
#include "../harith3.hpp"
#include "../truncation.hpp"
#include "../rand_trunc.hpp"
#include "../lanczostrunc.hpp"

#ifdef USE_COMPLEX
static field alpha = 1.0 + 1.0 * I;
#else
static field alpha = 1.0;
#endif

static uint problems = 0;

#ifdef USE_FLOAT
static real tolerance = 5.0e-6;
#else
static real tolerance = 1.0e-12;
#endif

#define IS_IN_RANGE(a, b, c) (((a) <= (b)) && ((b) <= (c)))

static void
check_triangularsolve(bool lower, bool unit, bool atrans,
                      pchmatrix a, bool xtrans, real tol)
{
  uint n = a->rc->size;

  pblock block;
  phmatrix xh, bh;
  real error, norm, eta;
  SVDTruncation trunc(tol);
  // RandomTruncation trunc(tol, 2, 0);
  // LanczosTruncation trunc(tol, 2);

  assert(n == a->cc->size);

  /****************************************************
   * hmatrix
   ****************************************************/

  eta = 1.0;

  block = build_nonstrict_block((pcluster)a->rc, (pcluster)a->cc, &eta,
                                admissible_max_cluster);

  xh = build_from_block_hmatrix(block, 0);
  random_hmatrix(xh, 8);

  bh = build_from_block_hmatrix(block, 0);
  clear_hmatrix(bh);

  if (xtrans)
  {
    addmul3_hmatrix(CONJ(alpha), false, xh, !atrans, a, trunc, false, bh);
  }
  else
  {
    addmul3_hmatrix(alpha, atrans, a, false, xh, trunc, false, bh);
  }

  triangularinvmul3_hmatrix(lower, unit, atrans, a, trunc, xtrans, bh);

  add_hmatrix(xtrans ? -CONJ(alpha) : -alpha, xh, 0, tol, bh);

  norm = norm2_hmatrix(xh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking hmatrix triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s)\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_hmatrix(xh, bh);
  triangularmul_hmatrix(lower, unit, atrans, a, 0, tol, xtrans, bh);

  triangularinvmul3_hmatrix(lower, unit, atrans, a, trunc, xtrans, bh);

  add_hmatrix(-1.0, xh, 0, tol, bh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking hmatrix triangulareval/triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s):\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  /****************************************************
   * hmatrix(amatrix)
   ****************************************************/

  random_hmatrix(xh, 8);

  del_hmatrix(bh);
  bh = new_full_hmatrix(a->cc, a->cc);
  clear_hmatrix(bh);

  if (xtrans)
  {
    addmul3_hmatrix(CONJ(alpha), false, xh, !atrans, a, trunc, false, bh);
  }
  else
  {
    addmul3_hmatrix(alpha, atrans, a, false, xh, trunc, false, bh);
  }

  triangularinvmul3_hmatrix(lower, unit, atrans, a, trunc, xtrans, bh);

  add_hmatrix(xtrans ? -CONJ(alpha) : -alpha, xh, 0, tol, bh);

  norm = norm2_hmatrix(xh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking hmatrix(amatrix) triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s)\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  clear_hmatrix(bh);
  add_hmatrix(1.0, xh, nullptr, tol, bh);
  triangularmul_hmatrix(lower, unit, atrans, a, 0, tol, xtrans, bh);

  triangularinvmul3_hmatrix(lower, unit, atrans, a, trunc, xtrans, bh);

  add_hmatrix(-1.0, xh, 0, tol, bh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking hmatrix(amatrix) triangulareval/triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s):\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  del_block(block);
  del_hmatrix(xh);
  del_hmatrix(bh);
}

static void
check_triangularsolve_amatrix(bool lower, bool unit, bool atrans,
                              pcamatrix a, pccluster root, bool xtrans,
                              real tol)
{
  uint n = a->rows;
  pblock block;
  phmatrix ah, xh, bh;
  real error, norm, eta;
  SVDTruncation trunc(tol);
  // RandomTruncation trunc(tol, 2, 0);
  // LanczosTruncation trunc(tol, 2);

  assert(n == a->cols);

  ah = (phmatrix)allocmem(sizeof(hmatrix));
  init_hmatrix(ah, root, root);
  ah->f = (pamatrix)a;
  update_hmatrix(ah);

  eta = 1.0;

  block = build_nonstrict_block((pcluster)root, (pcluster)root, &eta,
                                admissible_max_cluster);

  xh = build_from_block_hmatrix(block, 0);
  random_hmatrix(xh, 8);

  bh = build_from_block_hmatrix(block, 0);
  clear_hmatrix(bh);

  if (xtrans)
  {
    addmul3_hmatrix(CONJ(alpha), false, xh, !atrans, ah, trunc, false, bh);
  }
  else
  {
    addmul3_hmatrix(alpha, atrans, ah, false, xh, trunc, false, bh);
  }

  triangularinvmul_amatrix_hmatrix(lower, unit, atrans, a, nullptr, tol, xtrans,
                                   bh);

  add_hmatrix(xtrans ? -CONJ(alpha) : -alpha, xh, 0, tol, bh);

  norm = norm2_hmatrix(xh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking amatrix_hmatrix triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s)\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_hmatrix(xh, bh);
  triangularmul_hmatrix(lower, unit, atrans, ah, 0, tol, xtrans, bh);

  triangularinvmul_amatrix_hmatrix(lower, unit, atrans, a, nullptr, tol, xtrans,
                                   bh);

  add_hmatrix(-1.0, xh, 0, tol, bh);
  error = norm2_hmatrix(bh) / norm;

  (void)printf("Checking amatrix_hmatrix triangulareval/triangularsolve\n"
               "  (lower=%s, unit=%s, atrans=%s, xtrans=%s):\n"
               "  Accuracy %g, %sokay\n",
               (lower ? "tr" : "fl"),
               (unit ? "tr" : "fl"), (atrans ? "tr" : "fl"),
               (xtrans ? "tr" : "fl"), error,
               (IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT "));
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  del_block(block);
  del_hmatrix(xh);
  del_hmatrix(bh);
  freemem(ah);
}

static void
set_unit_hmatrix(phmatrix R)
{
  const uint sons = R->rsons;
  const uint minsons = UINT_MIN(sons, R->csons);

  uint i, m;

  if (R->son != nullptr)
  {
    for (i = 0; i < minsons; ++i)
    {
      set_unit_hmatrix(R->son[i + i * sons]);
    }
  }
  else
  {
    assert(R->f != nullptr);

    m = UINT_MIN(R->f->rows, R->f->cols);
    for (i = 0; i < m; ++i)
    {
      R->f->a[i + i * R->f->ld] = 1.0;
    }
  }
}

static void
set_unit_amatrix(pamatrix R)
{
  uint i, m;

  m = UINT_MIN(R->rows, R->cols);
  for (i = 0; i < m; ++i)
  {
    R->a[i + i * R->ld] = 1.0;
  }
}

int main(int argc, char **argv)
{
  phmatrix a, acopy, L, R;
  pamatrix La, Ra;
  pavector x, b, b2;
  uint n;
  real error;
  pcurve2d gr2;
  pbem2d bem2;
  pcluster root2;
  pblock block2;

  uint clf, m;
  real tol, eta, delta, eps_aca;

  n = 507;
  tol = tolerance;
  SVDTruncation trunc(tol);
  // RandomTruncation trunc(tol, 2, 0);
  // LanczosTruncation trunc(tol, 2);

  clf = 16;
  eta = 1.0;
  m = 4;
  delta = 1.0;
  eps_aca = tolerance;

  init_h2lib(&argc, &argv);

  gr2 = new_circle_curve2d(n, 0.333);
  bem2 = new_slp_laplace_bem2d(gr2, 2, BASIS_CONSTANT_BEM2D);
  root2 = build_bem2d_cluster(bem2, clf, BASIS_CONSTANT_BEM2D);
  block2 = build_nonstrict_block(root2, root2, &eta, admissible_max_cluster);
  setup_hmatrix_aprx_greenhybrid_row_bem2d(bem2, root2, root2, block2, m, 1,
                                           delta, eps_aca,
                                           build_bem2d_rect_quadpoints);
  setup_hmatrix_recomp_bem2d(bem2, true, eps_aca, true, eps_aca);

  (void)printf("----------------------------------------\n"
               "Check %u x %u LR factorization\n",
               n, n);

  (void)printf("Creating laplacebem2d SLP matrix\n");
  a = build_from_block_hmatrix(block2, 0);
  assemblecoarsen_bem2d_hmatrix(bem2, block2, a);

  (void)printf("Creating random solution and right-hand side\n");
  x = new_avector(n);
  random_avector(x);
  b = new_avector(n);
  b2 = new_avector(n);
  clear_avector(b);
  mvm_hmatrix_avector(alpha, false, a, x, b);
  copy_avector(b, b2);

  (void)printf("Copying matrix\n");
  acopy = clone_hmatrix(a);

  (void)printf("Computing LR factorization\n");
  lrdecomp3_hmatrix(a, trunc);

  (void)printf("Solving\n");
  lrsolve_hmatrix_avector(false, a, b);

  add_avector(-alpha, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  (void)printf("Solving adjoint\n");
  lrsolve_hmatrix_avector(true, a, b2);

  add_avector(-alpha, x, b2);
  error = norm2_avector(b2) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  (void)printf("Evaluating\n");
  copy_avector(x, b);
  triangulareval_hmatrix_avector(false, false, false, a, b);
  triangulareval_hmatrix_avector(true, true, false, a, b);
  mvm_hmatrix_avector(alpha, false, acopy, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, false, acopy, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_avector(x, b);
  triangulareval_hmatrix_avector(true, true, true, a, b);
  triangulareval_hmatrix_avector(false, false, true, a, b);
  mvm_hmatrix_avector(alpha, true, acopy, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, true, acopy, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  (void)printf("Building triangular factors\n");
  L = clone_lower_hmatrix(true, a);
  R = clone_upper_hmatrix(false, a);

  copy_avector(x, b);
  triangulareval_hmatrix_avector(false, false, false, a, b);
  mvm_hmatrix_avector(alpha, false, R, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, false, R, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_avector(x, b);
  triangulareval_hmatrix_avector(false, false, true, a, b);
  mvm_hmatrix_avector(alpha, true, R, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, true, R, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_avector(x, b);
  triangulareval_hmatrix_avector(true, true, false, a, b);
  mvm_hmatrix_avector(alpha, false, L, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, false, L, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  copy_avector(x, b);
  triangulareval_hmatrix_avector(true, true, true, a, b);
  mvm_hmatrix_avector(alpha, true, L, x, b);
  mvm_hmatrix_avector(-alpha - 1.0, true, L, x, b);
  error = norm2_avector(b) / norm2_avector(x);
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  (void)printf("Checking factorization\n");
  error = norm2_hmatrix(acopy);
  addmul3_hmatrix(alpha, false, L, false, R, trunc, false, acopy);
  addmul3_hmatrix(-alpha - 1.0, false, L, false, R, trunc, false, acopy);
  error = norm2_hmatrix(acopy) / error;
  (void)printf("  Accuracy %g, %sokay\n", error,
               IS_IN_RANGE(0.0, error, 10.0 * tol) ? "" : "    NOT ");
  if (!IS_IN_RANGE(0.0, error, 10.0 * tol))
    problems++;

  /* Check forward/backward substitution */
  (void)printf("----------------------------------------\n"
               "Checking lower triangular solve/eval\n");
  check_triangularsolve(true, true, false, L, false, tol);
  printf("\n");
  check_triangularsolve(true, true, false, L, true, tol);
  printf("\n");
  check_triangularsolve(true, true, true, L, false, tol);
  printf("\n");
  check_triangularsolve(true, true, true, L, true, tol);
  printf("\n");
  check_triangularsolve(true, false, false, L, false, tol);
  printf("\n");
  check_triangularsolve(true, false, false, L, true, tol);
  printf("\n");
  check_triangularsolve(true, false, true, L, false, tol);
  printf("\n");
  check_triangularsolve(true, false, true, L, true, tol);

  /* Check forward/backward substitution */
  (void)printf("----------------------------------------\n"
               "Checking lower triangular solve/eval amatrix_hmatrix\n");
  La = new_zero_amatrix(n, n);
  add_hmatrix_amatrix(1.0, false, L, La);
  check_triangularsolve_amatrix(true, true, false, La, L->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, true, false, La, L->rc, true, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, true, true, La, L->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, true, true, La, L->rc, true, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, false, false, La, L->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, false, false, La, L->rc, true, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, false, true, La, L->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(true, false, true, La, L->rc, true, tol);

  (void)printf("----------------------------------------\n"
               "Checking upper triangular solve/eval\n");
  check_triangularsolve(false, false, false, R, false, tol);
  printf("\n");
  check_triangularsolve(false, false, false, R, true, tol);
  printf("\n");
  check_triangularsolve(false, false, true, R, false, tol);
  printf("\n");
  check_triangularsolve(false, false, true, R, true, tol);
  printf("\n");
  set_unit_hmatrix(R);
  check_triangularsolve(false, true, false, R, false, tol);
  printf("\n");
  check_triangularsolve(false, true, false, R, true, tol);
  printf("\n");
  check_triangularsolve(false, true, true, R, false, tol);
  printf("\n");
  check_triangularsolve(false, true, true, R, true, tol);

  (void)printf("----------------------------------------\n"
               "Checking upper triangular solve/eval amatrix_hmatrix\n");
  Ra = new_zero_amatrix(n, n);
  add_hmatrix_amatrix(1.0, false, R, Ra);
  check_triangularsolve_amatrix(false, false, false, Ra, R->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, false, false, Ra, R->rc, true, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, false, true, Ra, R->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, false, true, Ra, R->rc, true, tol);
  printf("\n");
  set_unit_amatrix(Ra);
  check_triangularsolve_amatrix(false, true, false, Ra, R->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, true, false, Ra, R->rc, true, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, true, true, Ra, R->rc, false, tol);
  printf("\n");
  check_triangularsolve_amatrix(false, true, true, Ra, R->rc, true, tol);

  /* Final clean-up */
  (void)printf("Cleaning up\n");
  del_hmatrix(acopy);
  del_hmatrix(a);
  del_hmatrix(L);
  del_hmatrix(R);
  del_amatrix(La);
  del_amatrix(Ra);
  del_avector(b2);
  del_avector(b);
  del_avector(x);

  del_bem2d(bem2);
  del_block(block2);
  freemem(root2->idx);
  del_cluster(root2);
  del_curve2d(gr2);

  (void)printf("----------------------------------------\n"
               "  %u matrices and\n"
               "  %u vectors still active\n"
               "  %u errors found\n",
               getactives_amatrix(),
               getactives_avector(), problems);

  uninit_h2lib();

  return problems;
}
