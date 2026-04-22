#include "hlu.h"

#ifdef USE_CAIRO
#include <iostream>
#endif

#include <chrono>
#include <vector>
#include <algorithm>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

static void get_max_leaf_size_recursion(phmatrix h, std::vector<int>& v)
{
  if (h->f) {
    v.push_back(std::min(h->cc->size, h->rc->size));
  } else {
    for (uint i = 0; i < h->csons * h->rsons; i++) {
      get_max_leaf_size_recursion(h->son[i], v);
    }
  }
}

static int get_max_leaf_size(phmatrix h)
{
  std::vector<int> v;
  v.reserve(h->desc);
  get_max_leaf_size_recursion(h, v);
  
  return *std::max_element(v.begin(), v.end());
}

static void get_max_rank_recursion(phmatrix h, std::vector<int>& v)
{
  if (h->r) {
    v.push_back(h->r->k);
  } else {
    for (uint i = 0; i < h->csons * h->rsons; i++) {
      get_max_rank_recursion(h->son[i], v);
    }
  }
}

static int get_max_rank(phmatrix h)
{
  std::vector<int> v;
  v.reserve(h->desc);
  get_max_rank_recursion(h, v);
  
  return *std::max_element(v.begin(), v.end());
}

// helper function to set the block structures needed for subsequent H-LU
static void build_blocks_for_saddle_matrix(pcluster rootv, pcluster rootp, pcluster* rootrootv,
                              pblock* block_vel, pblock* block_grad, pblock* block_div, pblock* block_schur,
                              uint *idxv, const HLU_Options& opt)
{
  *rootrootv = new_cluster(rootv->size, idxv, 1, 3); // auxiliary cluster to make construction of blockb easier
  (*rootrootv)->son[0] = rootv;
  *block_vel = new_block(*rootrootv, *rootrootv, false, 1, 1);
  (*block_vel)->son[0] = build_nonstrict_block(rootv, rootv, opt.eta_vel, opt.adm_vel);
  update_block(*block_vel);

  *block_schur = new_block(rootp, rootp, false, 2, 2);
  (*block_schur)->son[0] = build_nonstrict_block(rootp->son[0], rootp->son[0], opt.eta_schur, opt.adm_schur);
  (*block_schur)->son[1] = new_block(rootp->son[1], rootp->son[0], false, 0, 0);
  (*block_schur)->son[2] = new_block(rootp->son[0], rootp->son[1], false, 0, 0);
  (*block_schur)->son[3] = new_block(rootp->son[1], rootp->son[1], false, 0, 0);
  update_block(*block_schur);

  *block_grad = new_block(rootp, *rootrootv, false, 2, 1);
  (*block_grad)->son[0] = build_nonstrict_block(rootp->son[0], rootv, opt.eta_grad, opt.adm_grad);
  (*block_grad)->son[1] = new_block(rootp->son[1], rootv, true, 0, 0);
  update_block(*block_grad);

  *block_div = new_block(rootp, *rootrootv, false, 2, 1);
  (*block_div)->son[0] = build_nonstrict_block(rootp->son[0], rootv, opt.eta_div, opt.adm_div);
  (*block_div)->son[1] = new_block(rootp->son[1], rootv, true, 0, 0);
  update_block(*block_div);
}

static void
pre_compute_csp_partition(pcblock b, uint bname, uint rname, uint cname,
		      uint pardepth, void *data)
{
  uint     *row, *col;
  uint    **cspdata;

  (void) bname;
  (void) pardepth;

  if (b->csons == 0 && b->rsons == 0) {
    cspdata = (uint **) data;
    row = cspdata[0];
    col = cspdata[1];

    row[rname]++;
    col[cname]++;
  }
}

uint
compute_csp_partition(pcblock b)
{
  uint      i, csp;
  uint     *row, *col;
  uint    **cspdata;

  row = (uint *) allocmem((size_t) sizeof(uint) * b->rc->desc);
  col = (uint *) allocmem((size_t) sizeof(uint) * b->cc->desc);
  cspdata = (uint **) allocmem((size_t) sizeof(uint *) * 2);
  cspdata[0] = row;
  cspdata[1] = col;

  for (i = 0; i < b->rc->desc; i++)
    row[i] = 0;
  for (i = 0; i < b->cc->desc; i++)
    col[i] = 0;

  iterate_block(b, 0, 0, 0, pre_compute_csp_partition, NULL, cspdata);

  csp = 0;
  for (i = 0; i < b->rc->desc; i++)
    csp = (row[i] > csp ? row[i] : csp);
  for (i = 0; i < b->cc->desc; i++)
    csp = (col[i] > csp ? col[i] : csp);

  freemem(row);
  freemem(col);
  freemem(cspdata);
  return csp;
}

Csp_Data compute_csp(pcluster rootv,
                     pcluster rootp,
                     const HLU_Options& opt)
{
  Csp_Data data;

  pcluster rootrootv = nullptr;
  pblock block_vel = nullptr, block_grad = nullptr, block_div = nullptr, block_schur = nullptr;
  uint *idxv;

  idxv = rootv->idx;
  
  // construct blocks such that pressure constraint is included
  build_blocks_for_saddle_matrix(rootv, rootp, &rootrootv,
                              &block_vel, &block_grad, &block_div, &block_schur,
                              idxv, opt);

  #ifdef USE_CAIRO
  cairo_t* cr_vel_block = new_cairopdf("vel_block.pdf", 2024.0, 2024.0);
  draw_cairo_block(cr_vel_block, block_vel, 0);
  cairo_destroy(cr_vel_block);
  std::cout << "cairo vel block pdf complete" << std::endl;
  #endif

  data.csp_vel = compute_csp_partition(block_vel->son[0]);
  data.csp_schur = compute_csp_partition(block_schur->son[0]);
  data.csp_grad = compute_csp_partition(block_grad->son[0]);
  data.csp_div = compute_csp_partition(block_div->son[0]);
  data.depth_vel = getdepth_block(block_vel);
  data.depth_grad = getdepth_block(block_grad);
  data.depth_div = getdepth_block(block_div);
  data.depth_schur = getdepth_block(block_schur);

  return data;
}

Block_HLU_Prcd::Block_HLU_Prcd(pcluster rootv,
                                pcluster rootp,
                                const HLU_Options& opt,
                                Block_HLU_Times& times,
                                HLU_Memory& memory,
                                HLU_Stats& stats,
                                BlockPrcdType prcd_type,
                                HArith harith,
                                std::unique_ptr<TruncationOperator>& trunc)
{
  phmatrix Ahat, Bhat1, Bhat2, Shat;
  pcluster rootrootv = nullptr;
  pblock block_vel = nullptr, block_grad = nullptr, block_div = nullptr, block_schur = nullptr;
  uint *idxv, *idxp;

  idxv = rootv->idx;
  idxp = rootp->idx;
  
  auto t_build_blocks_start = high_resolution_clock::now();
  
  // construct blocks such that pressure constraint is included
  build_blocks_for_saddle_matrix(rootv, rootp, &rootrootv,
                              &block_vel, &block_grad, &block_div, &block_schur,
                              idxv, opt);

  auto t_build_blocks_end = high_resolution_clock::now();
  auto t_build_hmatrix_start = high_resolution_clock::now();

  Ahat = build_from_block_hmatrix(block_vel, 0);
  copy_ordered_sparsematrix_to_hmatrix(opt.K->A, Ahat);

  Shat = build_from_block_hmatrix(block_schur, 0);
  copy_ordered_sparsematrix_to_hmatrix(opt.K->C, Shat);

  memory.vel = getsize_hmatrix(Ahat);
  stats.max_rank_vel = get_max_rank(Ahat);
  stats.max_leaf_size_vel = get_max_leaf_size(Ahat);

  auto t_build_hmatrix_end = high_resolution_clock::now();

  #ifdef USE_CAIRO
  cairo_t* cr_vel = new_cairopdf("vel_hmatrix.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel, Ahat, true, 0);
  cairo_destroy(cr_vel);
  cairo_t* cr_vel_01 = new_cairopdf("vel_hmatrix_01.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_01, Ahat, true, 3);
  cairo_destroy(cr_vel_01);
  cairo_t* cr_vel_02 = new_cairopdf("vel_hmatrix_02.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_02, Ahat, true, 4);
  cairo_destroy(cr_vel_02);
  cairo_t* cr_vel_03 = new_cairopdf("vel_hmatrix_03.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_03, Ahat, true, 5);
  cairo_destroy(cr_vel_03);
  cairo_t* cr_vel_04 = new_cairopdf("vel_hmatrix_04.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_04, Ahat, true, 6);
  cairo_destroy(cr_vel_04);
  cairo_t* cr_vel_05 = new_cairopdf("vel_hmatrix_05.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_05, Ahat, true, 7);
  cairo_destroy(cr_vel_05);
  std::cout << "cairo velocity pdf complete" << std::endl;
  #endif
  
  auto t_vel_lu_start = high_resolution_clock::now();
  if (harith == HArith::STANDARD)
    lrdecomp_hmatrix(Ahat, opt.tm, opt.eps_vel);
  if (harith == HArith::SUMEXP)
    lrdecomp3_hmatrix(Ahat, *trunc.get());
  auto t_vel_lu_end = high_resolution_clock::now();

  memory.vel_lu = getsize_hmatrix(Ahat);
  stats.max_rank_vel_lu = get_max_rank(Ahat);

  #ifdef USE_CAIRO
  cairo_t* cr_vel_lu = new_cairopdf("vel_lu_hmatrix.pdf", 1024.0, 1024.0);
  draw_cairo_hmatrix(cr_vel_lu, Ahat, true, 0);
  cairo_destroy(cr_vel_lu);
  std::cout << "cairo vel lu pdf complete" << std::endl;
  #endif

  auto t_schur_comp_start = high_resolution_clock::now();
  #ifndef USE_OPENMP
  Bhat1 = build_from_block_hmatrix(block_div, 0);
  Bhat2 = build_from_block_hmatrix(block_grad, 0);
  #endif
  #ifdef USE_OPENMP
  #pragma omp parallel for shared(Shat) private(Bhat1, Bhat2)
  #endif
  for (uint d = 0; d < 3; d++)
  {
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < 2; i++) {
      if (i == 0) {
        #ifdef USE_OPENMP
        Bhat1 = build_from_block_hmatrix(block_div, 0);
        #endif
        copy_ordered_sparsematrix_to_hmatrix(opt.K->B[d], Bhat1);
        memory.div = std::max(memory.div, getsize_hmatrix(Bhat1));
        stats.max_rank_div = std::max(stats.max_rank_div, get_max_rank(Bhat1));
        stats.max_leaf_size_div = std::max(stats.max_leaf_size_div, get_max_leaf_size(Bhat1));
        #ifdef USE_CAIRO
        if (d == 2) {
        cairo_t* cr_div = new_cairopdf("div_init_hmatrix.pdf", 1024.0, 1024.0*(static_cast<double>(rootp->size)/static_cast<double>(rootv->size)));
        draw_cairo_hmatrix(cr_div, Bhat1, true, 0);
        cairo_destroy(cr_div);
        std::cout << "cairo div init pdf complete" << std::endl;
        }
        #endif
        auto t_grad_upper_solve_start = high_resolution_clock::now();

        if (harith == HArith::STANDARD)
          triangularinvmul_hmatrix(false, false, true, Ahat, opt.tm, opt.eps_mul, true, Bhat1);
        if (harith == HArith::SUMEXP)
          triangularinvmul3_hmatrix(false, false, true, Ahat, *trunc.get(), true, Bhat1);

        auto t_grad_upper_solve_end = high_resolution_clock::now();
        duration<double> duration_grad_upper_solve = t_grad_upper_solve_end - t_grad_upper_solve_start;
        times.grad_upper_solve += duration_grad_upper_solve.count();
        memory.div_solved = getsize_hmatrix(Bhat1);
        stats.max_rank_div_solved = std::max(stats.max_rank_div_solved, get_max_rank(Bhat1));
      } else {
        #ifdef USE_OPENMP
        Bhat2 = build_from_block_hmatrix(block_grad, 0);
        #endif
        copy_ordered_sparsematrix_to_hmatrix(opt.K->B[d+3], Bhat2);
        memory.grad = std::max(memory.grad, getsize_hmatrix(Bhat2));
        stats.max_rank_grad = std::max(stats.max_rank_grad, get_max_rank(Bhat2));
        stats.max_leaf_size_grad = std::max(stats.max_leaf_size_grad, get_max_leaf_size(Bhat2));
        #ifdef USE_CAIRO
        if (d == 2) {
        cairo_t* cr_grad = new_cairopdf("grad_init_hmatrix.pdf", 1024.0, 1024.0*(static_cast<double>(rootp->size)/static_cast<double>(rootv->size)));
        draw_cairo_hmatrix(cr_grad, Bhat2, true, 0);
        cairo_destroy(cr_grad);
        std::cout << "cairo grad init pdf complete" << std::endl;
        }
        #endif
        auto t_grad_lower_solve_start = high_resolution_clock::now();

        if (harith == HArith::STANDARD)
          triangularinvmul_hmatrix(true, true, false, Ahat, opt.tm, opt.eps_mul, true, Bhat2);
        if (harith == HArith::SUMEXP)
          triangularinvmul3_hmatrix(true, true, false, Ahat, *trunc.get(), true, Bhat2);
        
        auto t_grad_lower_solve_end = high_resolution_clock::now();
        duration<double> duration_grad_lower_solve = t_grad_lower_solve_end - t_grad_lower_solve_start;
        times.grad_lower_solve += duration_grad_lower_solve.count();
        memory.grad_solved = getsize_hmatrix(Bhat2);
        stats.max_rank_grad_solved = std::max(stats.max_rank_grad_solved, get_max_rank(Bhat2));
      }
    }

    auto t_mult_start = high_resolution_clock::now();
    #ifdef USE_OPENMP
    phmatrix Temp;
    Temp = clonestructure_hmatrix(Shat);
    clear_hmatrix(Temp);
    addmul_hmatrix(-1.0, false, Bhat1, true, Bhat2, opt.tm, opt.eps_mul, Temp);

    #pragma omp critical
    {
      add_hmatrix(1.0, Temp, opt.tm, opt.eps_mul, Shat);
    }
    del_hmatrix(Temp);
    #else // serial implementation
    if (harith == HArith::STANDARD)
      addmul_hmatrix(-1.0, false, Bhat1, true, Bhat2, opt.tm, opt.eps_mul, Shat);
    if (harith == HArith::SUMEXP)
      addmul3_hmatrix(-1.0, false, Bhat1, true, Bhat2, *trunc.get(), false, Shat);
    #endif
    auto t_mult_end = high_resolution_clock::now();
    duration<double> duration_mult = t_mult_end - t_mult_start;
    times.grad_schur_multiplication += duration_mult.count();

    #ifdef USE_CAIRO
    if (d == 2) {
    cairo_t* cr_div = new_cairopdf("div_hmatrix.pdf", 1024.0, 1024.0*(static_cast<double>(rootp->size)/static_cast<double>(rootv->size)));
    draw_cairo_hmatrix(cr_div, Bhat1, true, 0);
    cairo_destroy(cr_div);
    std::cout << "cairo div pdf complete" << std::endl;

    cairo_t* cr_grad = new_cairopdf("grad_hmatrix.pdf", 1024.0, 1024.0*(static_cast<double>(rootp->size)/static_cast<double>(rootv->size)));
    draw_cairo_hmatrix(cr_grad, Bhat2, true, 0);
    cairo_destroy(cr_grad);
    std::cout << "cairo grad pdf complete" << std::endl;
    }
    #endif
    #ifdef USE_OPENMP
    del_hmatrix(Bhat1);
    del_hmatrix(Bhat2);
    #endif
  }
  #ifndef USE_OPENMP
  del_hmatrix(Bhat1);
  del_hmatrix(Bhat2);
  #endif

  // Set total times for the schur complement computation
  // times->schur_computation = times->grad_schur_multiplication + times->grad_lower_solve + times->grad_upper_solve;
  auto t_schur_comp_end = high_resolution_clock::now();

  memory.schur = getsize_hmatrix(Shat);
  stats.max_rank_schur = get_max_rank(Shat);
  stats.max_leaf_size_schur = get_max_leaf_size(Shat);

  #ifdef USE_CAIRO
  cairo_t* cr_schur = new_cairopdf("schur_hmatrix.pdf", 512.0, 512.0);
  draw_cairo_hmatrix(cr_schur, Shat, true, 0);
  cairo_destroy(cr_schur);
  std::cout << "cairo schur pdf complete" << std::endl;
  #endif

  auto t_schur_lu_start = high_resolution_clock::now();
  if (harith == HArith::STANDARD)
    lrdecomp_hmatrix(Shat, opt.tm, opt.eps_schur);
  if (harith == HArith::SUMEXP)
    lrdecomp3_hmatrix(Shat, *trunc.get());
  auto t_schur_lu_end = high_resolution_clock::now();

  memory.schur_lu = getsize_hmatrix(Shat);
  stats.max_rank_schur_lu = get_max_rank(Shat);

  #ifdef USE_CAIRO
  cairo_t* cr_schur_lu = new_cairopdf("schur_lu_hmatrix.pdf", 512.0, 512.0);
  draw_cairo_hmatrix(cr_schur_lu, Shat, true, 0);
  cairo_destroy(cr_schur_lu);
  std::cout << "cairo schur lu pdf complete" << std::endl;
  #endif

  duration<double> duration_build_blocks = t_build_blocks_end - t_build_blocks_start;
  times.build_blocks += duration_build_blocks.count();

  duration<double> duration_build_hmatrix = t_build_hmatrix_end - t_build_hmatrix_start;
  times.build_hmatrix += duration_build_hmatrix.count();

  duration<double> duration_vel_lu = t_vel_lu_end - t_vel_lu_start;
  times.velocity_lu += duration_vel_lu.count();

  duration<double> duration_schur_comp = t_schur_comp_end - t_schur_comp_start;
  times.schur_computation += duration_schur_comp.count();

  duration<double> duration_schur_lu = t_schur_lu_end - t_schur_lu_start;
  times.schur_lu += duration_schur_lu.count();

  this->A = Ahat;
  this->S = Shat;
  this->B = opt.K->B;
  this->C = opt.K->C;
  this->block_vel = block_vel;
  this->block_grad = block_grad;
  this->block_div = block_div;
  this->block_schur = block_schur;
  this->idxu = idxv;
  this->idxp = idxp;
  this->root_velocity = rootrootv;
  this->root_pressure = rootp;
  this->prcd_type = prcd_type;  
}

Block_HLU_Prcd::~Block_HLU_Prcd()
{

  del_block(this->block_vel);
  del_block(this->block_schur);
  del_block(this->block_grad);
  del_block(this->block_div);

  del_cluster(this->root_velocity);
  del_cluster(this->root_pressure);

  delete[] this->idxu;
  delete[] this->idxp;

  del_hmatrix(this->A);
  del_hmatrix(this->S);
  
}

void Block_HLU_Prcd::apply_preconditioner(pavector r)
{

  pavector *r1, r2;
  uint n, m, d;

  r1 = new pavector[3];

  n = this->A->rc->size;
  m = this->S->rc->size;

  assert(3 * n + m == r->dim);

  for (d = 0; d < 3; d++)
    r1[d] = new_sub_avector(r, n, d * n);
  r2 = new_sub_avector(r, m, 3 * n);

  /* r1 <- A^{-1}*r1 */
  for (d = 0; d < 3; d++)
    lrsolve_hmatrix_avector(false, this->A, r1[d]);

  // Apply lower triangular part
  if (this->prcd_type == BLOCK_TRIANGULAR)
  {
    for(d = 0; d < 3; d++)
      addeval_sparsematrix_avector(-1.0, this->B[d], r1[d], r2);
  }

  /* r2 <- S^{-1}*r2 */
  lrsolve_hmatrix_avector(false, this->S, r2);

  for (d = 0; d < 3; d++)
    del_avector(r1[d]);

  delete[] r1;

  del_avector(r2);
}

void mvm_schurcomplement(field alpha, bool trans, Block_HLU_Prcd *P, pavector x, pavector y)
{
  pavector tmp;
  tmp = new_zero_avector(P->A->rc->size);

  for(uint d = 0; d < 3; d++)
  {
    clear_avector(tmp);
    mvm_sparsematrix_avector(1.0, true, P->B[d], x, tmp);
    lrsolve_hmatrix_avector(trans, P->A, tmp);
    mvm_sparsematrix_avector(-alpha, false, P->B[d], tmp, y);
  }

  del_avector(tmp);
}

void mvm_schurcomplement_rbffd(field alpha, bool trans, Block_HLU_Prcd *P, pavector x, pavector y)
{
  pavector tmp;
  tmp = new_zero_avector(P->A->rc->size);

  if (trans)
  {
    for(uint d = 0; d < 3; d++)
    {
      clear_avector(tmp);
      mvm_sparsematrix_avector(1.0, true, P->B[d], x, tmp);
      lrsolve_hmatrix_avector(trans, P->A, tmp);
      mvm_sparsematrix_avector(-alpha, false, P->B[d+3], tmp, y);
    }
    mvm_sparsematrix_avector(1.0, true, P->C, x, y);
  } else {
    for(uint d = 0; d < 3; d++)
    {
      clear_avector(tmp);
      mvm_sparsematrix_avector(1.0, true, P->B[d+3], x, tmp);
      lrsolve_hmatrix_avector(trans, P->A, tmp);
      mvm_sparsematrix_avector(-alpha, false, P->B[d], tmp, y);
    }
    mvm_sparsematrix_avector(1.0, false, P->C, x, y);
  }
  del_avector(tmp);
}

real
norm2diff_lr_schurcomplement_hmatrix(Block_HLU_Prcd *P, pchmatrix LR)
{
  return norm2diff_pre_matrix((mvm_t) mvm_schurcomplement, (void *) P,
			      (prcd_t) lreval_n_hmatrix_avector,
			      (prcd_t) lreval_t_hmatrix_avector, (void *) LR,
			      P->S->rc->size, P->S->cc->size);
}

real
norm2diff_lr_schurcomplement_hmatrix_rbffd(Block_HLU_Prcd *P, pchmatrix LR)
{
  return norm2diff_pre_matrix((mvm_t) mvm_schurcomplement_rbffd, (void *) P,
			      (prcd_t) lreval_n_hmatrix_avector,
			      (prcd_t) lreval_t_hmatrix_avector, (void *) LR,
			      P->S->rc->size, P->S->cc->size);
}

real
norm2diff_id_lr_schurcomplement_hmatrix_rbffd(Block_HLU_Prcd *P, pchmatrix LR)
{
  return norm2diff_id_pre_matrix((mvm_t) mvm_schurcomplement_rbffd, (void *) P,
            (prcd_t) lreval_n_hmatrix_avector,
            (prcd_t) lreval_t_hmatrix_avector, (void *) LR,
            P->S->rc->size, P->S->cc->size);
}

static void eval_n_tria_prcd_avector(Block_HLU_Prcd *P, pavector r)
{
  pavector *r1, r2;
  uint n, m, d;

  r1 = new pavector[3];

  n = P->A->rc->size;
  m = P->S->rc->size;

  assert(3 * n + m == r->dim);

  for (d = 0; d < 3; d++)
    r1[d] = new_sub_avector(r, n, d * n);
  r2 = new_sub_avector(r, m, 3 * n);

  /* r1 <- A^{-1}*r1 */
  for (d = 0; d < 3; d++)
    lrsolve_hmatrix_avector(false, P->A, r1[d]);

  // Apply lower triangular part
  if (P->prcd_type == BLOCK_TRIANGULAR)
  {
    for(d = 0; d < 3; d++)
      addeval_sparsematrix_avector(-1.0, P->B[d], r1[d], r2);
  }

  /* r2 <- S^{-1}*r2 */
  lrsolve_hmatrix_avector(false, P->S, r2);

  for (d = 0; d < 3; d++)
    del_avector(r1[d]);

  delete[] r1;

  del_avector(r2);
}

static void eval_t_tria_prcd_avector(Block_HLU_Prcd *P, pavector r)
{
  pavector *r1, r2;
  uint n, m, d;

  r1 = new pavector[3];

  n = P->A->rc->size;
  m = P->S->rc->size;

  assert(3 * n + m == r->dim);

  for (d = 0; d < 3; d++)
    r1[d] = new_sub_avector(r, n, d * n);
  r2 = new_sub_avector(r, m, 3 * n);

  /* r2 <- S^{-1}*r2 */
  lrsolve_hmatrix_avector(true, P->S, r2);
  
  // Apply lower triangular part
  if (P->prcd_type == BLOCK_TRIANGULAR)
  {
    for(d = 0; d < 3; d++)
      addevaltrans_sparsematrix_avector(-1.0, P->B[d], r2, r1[d]);
  }

  /* r1 <- A^{-1}*r1 */
  for (d = 0; d < 3; d++)
    lrsolve_hmatrix_avector(true, P->A, r1[d]);

  for (d = 0; d < 3; d++)
    del_avector(r1[d]);

  delete[] r1;

  del_avector(r2);
}

real
norm2diff_id_tria_prcd(Block_HLU_Prcd *P, psparsematrix A)
{
  return norm2diff_id_pre_matrix((mvm_t) mvm_sparsematrix_avector, (void *) A,
            (prcd_t) eval_n_tria_prcd_avector,
            (prcd_t) eval_t_tria_prcd_avector, (void *) P,
            3*P->A->rc->size + P->S->rc->size, 3*P->A->cc->size + P->S->cc->size);
}