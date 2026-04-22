#include "../../../H2Lib/h2lib.h"
#include "../harith3.hpp"
#include "../truncation.hpp"
#include "../rand_trunc.hpp"

uint *get_indexset(uint n, uint div)
{
  uint *idx = new uint[n];
  uint *i = new uint[div];
  uint mod = n % div;
  uint ndiv = n / div;
  uint *off = new uint[div];

  off[0] = 0;
  i[0] = 0;
  for (uint j = 1; j < div; j++)
  {
    off[j] = (mod >= j ? j*(n + 4 - mod) / div : j*ndiv);
    i[j] = 0;
  }

  for (uint k = 0; k < n; k++)
  {
    uint j;
    for (j = 0; j < div - 1; j++)
      if (k % div == j)
        break;
    assert(j < div);

    idx[off[j] + i[j]] = k;
    i[j] = i[j] + 1;
  }

  delete[] off;
  delete[] i;

  return idx;
}

int check_addmul3()
{
  phmatrix x, y, z, zhat;
  pcluster root1, root2, root3;
  uint *idx1, *idx2, *idx3;

  idx1 = get_indexset(10, 2);
  idx2 = get_indexset(20, 5);
  idx3 = get_indexset(15, 3);

  // Cluster trees
  root1 = new_cluster(10, idx1, 2, 0);
  root1->son[0] = new_cluster(5, idx1, 0, 0);
  root1->son[1] = new_cluster(5, idx1 + 5, 0, 0);

  root2 = new_cluster(20, idx2, 2, 0);
  root2->son[0] = new_cluster(8, idx2, 2, 0);
  root2->son[1] = new_cluster(12, idx2 + 8, 2, 0);
  root2->son[0]->son[0] = new_cluster(4, idx2, 0, 0);
  root2->son[0]->son[1] = new_cluster(4, idx2 + 4, 0, 0);
  root2->son[1]->son[0] = new_cluster(4, idx2 + 8, 0, 0);
  root2->son[1]->son[1] = new_cluster(8, idx2 + 12, 2, 0);
  root2->son[1]->son[1]->son[0] = new_cluster(4, idx2 + 12, 0, 0);
  root2->son[1]->son[1]->son[1] = new_cluster(4, idx2 + 16, 0, 0);

  root3 = new_cluster(15, idx3, 3, 0);
  root3->son[0] = new_cluster(5, idx3, 0, 0);
  root3->son[1] = new_cluster(5, idx3, 0, 0);
  root3->son[2] = new_cluster(5, idx3, 0, 0);

  pblock blockx, blocky, blockz;
  // Build block for matrix x
  blockx = new_block(root1, root2, false, 2, 2);
  blockx->son[0] = new_block(root1->son[0], root2->son[0], false, 1, 2);
  blockx->son[1] = new_block(root1->son[1], root2->son[0], false, 0, 0);
  blockx->son[2] = new_block(root1->son[0], root2->son[1], true, 0, 0);
  blockx->son[3] = new_block(root1->son[1], root2->son[1], false, 1, 2);
  blockx->son[0]->son[0] = new_block(root1->son[0], root2->son[0]->son[0], false, 0, 0);
  blockx->son[0]->son[1] = new_block(root1->son[0], root2->son[0]->son[1], false, 0, 0);
  blockx->son[3]->son[0] = new_block(root1->son[1], root2->son[1]->son[0], false, 0, 0);
  blockx->son[3]->son[1] = new_block(root1->son[1], root2->son[1]->son[1], true, 0, 0);
  // Build block for matrix y
  blocky = new_block(root2, root3, false, 2, 3);
  blocky->son[0] = new_block(root2->son[0], root3->son[0], false, 0, 0);
  blocky->son[1] = new_block(root2->son[1], root3->son[0], false, 2, 1);
  blocky->son[2] = new_block(root2->son[0], root3->son[1], false, 0, 0);
  blocky->son[3] = new_block(root2->son[1], root3->son[1], true, 0, 0);
  blocky->son[4] = new_block(root2->son[0], root3->son[2], false, 0, 0);
  blocky->son[5] = new_block(root2->son[1], root3->son[2], true, 0, 0);
  blocky->son[1]->son[0] = new_block(root2->son[1]->son[0], root3->son[0], false, 0, 0);
  blocky->son[1]->son[1] = new_block(root2->son[1]->son[1], root3->son[0], true, 0, 0);
  // Build block for matrix z
  blockz = new_block(root1, root3, false, 2, 3);
  blockz->son[0] = new_block(root1->son[0], root3->son[0], false, 0, 0);
  blockz->son[1] = new_block(root1->son[1], root3->son[0], true, 0, 0);
  blockz->son[2] = new_block(root1->son[0], root3->son[1], true, 0, 0);
  blockz->son[3] = new_block(root1->son[1], root3->son[1], false, 0, 0);
  blockz->son[4] = new_block(root1->son[0], root3->son[2], true, 0, 0);
  blockz->son[5] = new_block(root1->son[1], root3->son[2], true, 0, 0);

  // Build H-matrices
  uint k = 3;
  x = build_from_block_hmatrix(blockx, k);
  y = build_from_block_hmatrix(blocky, k);
  z = build_from_block_hmatrix(blockz, k);
  zhat = build_from_block_hmatrix(blockz, k);

  random_hmatrix(x, k);
  random_hmatrix(y, k);
  clear_hmatrix(z);
  clear_hmatrix(zhat);

  // Compute standard H-matrix product
  ptruncmode tm = new_releucl_truncmode();
  real eps = 0.0;
  addmul_hmatrix(1.0, false, x, false, y, tm, eps, z);
  del_truncmode(tm);

  // Compute new product
  // SVDTruncation trunc(eps);
  RandomTruncation trunc(eps, 1);

  addmul3_hmatrix(1.0, false, x, false, y, trunc, false, zhat);

  real norm = norm2diff_hmatrix(z, zhat);
  printf("Difference on spectral norm: %3.2e\n", norm);

  /* Clean up*/
  del_hmatrix(x);
  del_hmatrix(y);
  del_hmatrix(z);
  del_hmatrix(zhat);
  del_block(blockx);
  del_block(blocky);
  del_block(blockz);
  del_cluster(root1);
  del_cluster(root2);
  del_cluster(root3);
  delete[] idx1;
  delete[] idx2;
  delete[] idx3;

  return EXIT_SUCCESS;
}

int check_addmul3_dense()
{
  phmatrix x, y, z, zhat;
  pcluster root1, root2, root3;
  uint *idx1, *idx2, *idx3;

  idx1 = get_indexset(10, 2);
  idx2 = get_indexset(20, 5);
  idx3 = get_indexset(15, 3);

  // Cluster trees
  root1 = new_cluster(10, idx1, 2, 0);
  root1->son[0] = new_cluster(5, idx1, 0, 0);
  root1->son[1] = new_cluster(5, idx1 + 5, 0, 0);

  root2 = new_cluster(20, idx2, 2, 0);
  root2->son[0] = new_cluster(8, idx2, 2, 0);
  root2->son[1] = new_cluster(12, idx2 + 8, 2, 0);
  root2->son[0]->son[0] = new_cluster(4, idx2, 0, 0);
  root2->son[0]->son[1] = new_cluster(4, idx2 + 4, 0, 0);
  root2->son[1]->son[0] = new_cluster(4, idx2 + 8, 0, 0);
  root2->son[1]->son[1] = new_cluster(8, idx2 + 12, 2, 0);
  root2->son[1]->son[1]->son[0] = new_cluster(4, idx2 + 12, 0, 0);
  root2->son[1]->son[1]->son[1] = new_cluster(4, idx2 + 16, 0, 0);

  root3 = new_cluster(15, idx3, 3, 0);
  root3->son[0] = new_cluster(5, idx3, 0, 0);
  root3->son[1] = new_cluster(5, idx3, 0, 0);
  root3->son[2] = new_cluster(5, idx3, 0, 0);

  pblock blockx, blocky;
  // Build block for matrix x
  blockx = new_block(root1, root2, false, 2, 2);
  blockx->son[0] = new_block(root1->son[0], root2->son[0], false, 1, 2);
  blockx->son[1] = new_block(root1->son[1], root2->son[0], false, 0, 0);
  blockx->son[2] = new_block(root1->son[0], root2->son[1], true, 0, 0);
  blockx->son[3] = new_block(root1->son[1], root2->son[1], false, 1, 2);
  blockx->son[0]->son[0] = new_block(root1->son[0], root2->son[0]->son[0], false, 0, 0);
  blockx->son[0]->son[1] = new_block(root1->son[0], root2->son[0]->son[1], false, 0, 0);
  blockx->son[3]->son[0] = new_block(root1->son[1], root2->son[1]->son[0], false, 0, 0);
  blockx->son[3]->son[1] = new_block(root1->son[1], root2->son[1]->son[1], true, 0, 0);
  // Build block for matrix y
  blocky = new_block(root2, root3, false, 2, 3);
  blocky->son[0] = new_block(root2->son[0], root3->son[0], false, 0, 0);
  blocky->son[1] = new_block(root2->son[1], root3->son[0], false, 2, 1);
  blocky->son[2] = new_block(root2->son[0], root3->son[1], false, 0, 0);
  blocky->son[3] = new_block(root2->son[1], root3->son[1], true, 0, 0);
  blocky->son[4] = new_block(root2->son[0], root3->son[2], false, 0, 0);
  blocky->son[5] = new_block(root2->son[1], root3->son[2], true, 0, 0);
  blocky->son[1]->son[0] = new_block(root2->son[1]->son[0], root3->son[0], false, 0, 0);
  blocky->son[1]->son[1] = new_block(root2->son[1]->son[1], root3->son[0], true, 0, 0);

  // Build H-matrices
  uint k = 3;
  x = build_from_block_hmatrix(blockx, k);
  y = build_from_block_hmatrix(blocky, k);
  z = new_full_hmatrix(root1, root3);
  zhat = new_full_hmatrix(root1, root3);

  random_hmatrix(x, k);
  random_hmatrix(y, k);
  clear_hmatrix(z);
  clear_hmatrix(zhat);

  // Compute standard H-matrix product
  ptruncmode tm = new_releucl_truncmode();
  real eps = 0.0;
  addmul_hmatrix(1.0, false, x, false, y, tm, eps, z);

  // Compute new product
  SVDTruncation trunc(eps, tm);
  // AdaptiveRandomTruncation *trunc = new AdaptiveRandomTruncation(eps);

  addmul3_hmatrix(1.0, false, x, false, y, trunc, false, zhat);

  real norm = norm2diff_hmatrix(z, zhat);
  printf("Difference on spectral norm: %3.2e\n", norm);

  /* Clean up*/
  del_hmatrix(x);
  del_hmatrix(y);
  del_hmatrix(z);
  del_hmatrix(zhat);
  del_block(blockx);
  del_block(blocky);
  del_cluster(root1);
  del_cluster(root2);
  del_cluster(root3);
  delete[] idx1;
  delete[] idx2;
  delete[] idx3;

  return EXIT_SUCCESS;
}

int check2_addmul3()
{
  uint rfs = 4;
  ptri2d *gr = new ptri2d[rfs+1];

  gr[0] = new_lshape_tri2d();
  for(uint i = 1; i <= rfs; i++)
    gr[i] = refine_tri2d(gr[i-1], 0);

  ptri2dp1 fem = new_tri2dp1(gr[rfs]);
  psparsematrix A = build_tri2dp1_sparsematrix(fem);
  assemble_tri2dp1_laplace_sparsematrix(fem, A, 0);

  uint *idx = new uint[fem->ndof];
  pclustergeometry cg = build_tri2dp1_clustergeometry(fem, idx);

  uint *flag = new uint[fem->ndof];
  for(uint i = 0; i < fem->ndof; i++)
    flag[i] = 0;

  pcluster root = build_adaptive_dd_cluster(cg, fem->ndof, idx, 20, A, 2, flag);
  real eta = 1.0;
  pblock b = build_nonstrict_block(root, root, &eta, admissible_dd_cluster);

  phmatrix LR = build_from_block_hmatrix(b, 0);
  copy_sparsematrix_hmatrix(A, LR);

  ptruncmode tm = new_releucl_truncmode();
  real eps = 0.0;

  lrdecomp_hmatrix(LR, tm, eps);
  del_truncmode(tm);

  phmatrix L = clone_lower_hmatrix(true, LR);
  phmatrix R = clone_upper_hmatrix(false, LR);

  phmatrix B = clonestructure_hmatrix(LR);
  // SVDTruncation trunc(eps, tm);
  RandomTruncation trunc(eps, 1);
  // AdaptiveRandomTruncation *trunc = new AdaptiveRandomTruncation(eps);

  addmul3_hmatrix(1.0, false, L, false, R, trunc, false, B);

  real norm = norm2diff_sparsematrix_hmatrix(B, A);

  printf("Not transposed: Difference on spectral norm: %3.2e\n", norm);

  clear_hmatrix(B);
  addmul3_hmatrix(1.0, true, R, true, L, trunc, false, B);

  norm = norm2diff_sparsematrix_hmatrix(B, A);

  printf("Transposed: Difference on spectral norm: %3.2e\n", norm);

  del_hmatrix(B);
  del_hmatrix(L);
  del_hmatrix(R);
  del_hmatrix(LR);
  del_block(b);
  del_cluster(root);
  delete [] idx;
  delete [] flag;
  del_sparsematrix(A);
  del_tri2dp1(fem);
  for(uint i = 0; i <= rfs; i++)
    del_tri2d(gr[i]);
  delete [] gr;

  return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
  init_h2lib(&argc, &argv);

  check_addmul3();

  check_addmul3_dense();

  check2_addmul3();

  uninit_h2lib();
}