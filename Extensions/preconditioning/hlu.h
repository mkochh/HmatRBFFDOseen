#ifndef HLU_HEADER
#define HLU_HEADER

#include "../../H2Lib/h2lib.h"
#include "prec.h"
#include "../harith/harith3.hpp"

/**
 * @brief Collection of parameters to build up the H-Matrices
 *        for an Block_HLU_Prec preconditioner
 *
 */
struct HLU_Options {
  pspmatrix K; // Saddle point system matrix

  // Admissibility parameters
  void *eta_vel; // Admissibility parameter for the velocity matrix
  void *eta_grad; // Admissibility parameter for the gradient matrix
  void *eta_div; // Admissibility parameter for the divergence matrix
  void *eta_schur; // Admissibility parameter for the Schur complement

  admissible adm_vel;  // Admissibility condition for the Laplace/Convection part
  admissible adm_grad; // Admissibility condition for the grad matrix
  admissible adm_div; // Admissibility condition for the div matrix
  admissible adm_schur;  // Admissibility condition for the Schurcomplement

  ptruncmode tm;  // Truncation mode to use
  real eps_vel;   // Truncation accuracy for the velocity block
  real eps_mul;   // Truncation accuracy for the Schur complement computation
  real eps_schur; // Truncation accuracy for the Schur complement
};

struct Block_HLU_Times {
  real build_blocks = 0.0;
  real build_hmatrix = 0.0;

  real velocity_lu = 0.0;

  real grad_upper_solve = 0.0;
  real grad_lower_solve = 0.0;
  real grad_schur_multiplication = 0.0;
  real schur_computation = 0.0;

  real schur_lu = 0.0;
};

struct HLU_Memory {
  size_t vel = 0.0;
  size_t vel_lu = 0.0;
  size_t grad = 0.0;
  size_t grad_solved = 0.0;
  size_t div = 0.0;
  size_t div_solved = 0.0;
  size_t schur = 0.0;
  size_t schur_lu = 0.0;
};

struct HLU_Stats {
  int max_rank_vel = 0;
  int max_leaf_size_vel = 0;
  int max_rank_vel_lu = 0;
  int max_rank_grad = 0;
  int max_leaf_size_grad = 0;
  int max_rank_grad_solved = 0;
  int max_rank_div = 0;
  int max_leaf_size_div = 0;
  int max_rank_div_solved = 0;
  int max_rank_schur = 0;
  int max_leaf_size_schur = 0;
  int max_rank_schur_lu = 0;
};

struct Csp_Data {
  uint csp_vel = 0;
  uint csp_grad = 0;
  uint csp_div = 0;
  uint csp_schur = 0;
  uint depth_vel = 0;
  uint depth_grad = 0;
  uint depth_div = 0;
  uint depth_schur = 0;
};

enum BlockPrcdType {
  BLOCK_DIAGONAL,
  BLOCK_TRIANGULAR
};

enum class HArith {
  STANDARD,
  SUMEXP
};

class Block_HLU_Prcd : public Preconditioner {
public:
  phmatrix A;
  phmatrix S;
  psparsematrix *B;
  psparsematrix C;

  pcluster root_velocity;
  pcluster root_pressure;
  pblock block_vel;
  pblock block_schur;
  pblock block_grad;
  pblock block_div;
  uint *idxu;
  uint *idxp;

  BlockPrcdType prcd_type;

  /**
   * @brief Construct a new Block_HLU_Prec preconditioner and measure
   * construction timings
   */

  Block_HLU_Prcd(pcluster rootv,
                  pcluster rootp,
                  const HLU_Options& opt,
                  Block_HLU_Times& times,
                  HLU_Memory& memory,
                  HLU_Stats& stats,
                  BlockPrcdType prcd_type,
                  HArith harith,
                  std::unique_ptr<TruncationOperator>& trunc);

  ~Block_HLU_Prcd();

  /**
   * @brief Apply a block preconditioner to a given vector
   *
   * @param P Pointer to used preconditioner
   * @param r Vector to apply the preconditioner on
   */
  void apply_preconditioner(pavector r);
};

uint compute_csp_partition(pcblock b);

Csp_Data compute_csp(pcluster rootv, pcluster rootp, const HLU_Options& opt);

/* Matrix vector multiplicatio with the (approximate) schur complement for lu accuracy */
void mvm_schurcomplement(field alpha, bool trans, Block_HLU_Prcd *P, pavector x, pavector y);

void mvm_schurcomplement_rbffd(field alpha, bool trans, Block_HLU_Prcd *P, pavector x, pavector y);

real norm2diff_lr_schurcomplement_hmatrix(Block_HLU_Prcd *P, pchmatrix LR);

real norm2diff_lr_schurcomplement_hmatrix_rbffd(Block_HLU_Prcd *P, pchmatrix LR);

real norm2diff_id_lr_schurcomplement_hmatrix_rbffd(Block_HLU_Prcd *P, pchmatrix LR);

real norm2diff_id_tria_prcd(Block_HLU_Prcd *P, psparsematrix A);

#endif