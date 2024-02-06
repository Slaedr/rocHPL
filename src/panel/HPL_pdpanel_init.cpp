/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include "hpl_hip.hpp"

static int hostMalloc(void** ptr, const size_t bytes) {
  hipError_t err = hipHostMalloc(ptr, bytes);

  /*Check workspace allocation is valid*/
  if(err != hipSuccess) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

static int deviceMalloc(void** ptr, const size_t bytes) {
  hipError_t err = hipMalloc(ptr, bytes);

  /*Check workspace allocation is valid*/
  if(err != hipSuccess) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

void HPL_pdpanel_init(HPL_T_grid*  GRID,
                      HPL_T_palg*  ALGO,
                      const int    M,
                      const int    N,
                      const int    JB,
                      HPL_T_pmat*  A,
                      const int    IA,
                      const int    JA,
                      const int    TAG,
                      HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_init initializes a panel data structure.
   *
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * M       (local input)                 const int
   *         On entry, M specifies the global number of rows of the panel.
   *         M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  global number of columns of the
   *         panel and trailing submatrix. N must be at least zero.
   *
   * JB      (global input)                const int
   *         On entry, JB specifies is the number of columns of the panel.
   *         JB must be at least zero.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * IA      (global input)                const int
   *         On entry,  IA  is  the global row index identifying the panel
   *         and trailing submatrix. IA must be at least zero.
   *
   * JA      (global input)                const int
   *         On entry, JA is the global column index identifying the panel
   *         and trailing submatrix. JA must be at least zero.
   *
   * TAG     (global input)                const int
   *         On entry, TAG is the row broadcast message id.
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  int icurcol, icurrow, ii, jj;

  PANEL->grid = GRID; /* ptr to the process grid */
  PANEL->algo = ALGO; /* ptr to the algo parameters */
  PANEL->pmat = A;    /* ptr to the local array info */

  const int myrow = GRID->myrow;
  const int mycol = GRID->mycol;
  const int nprow = GRID->nprow;
  const int npcol = GRID->npcol;
  const int nb    = A->nb;

  HPL_infog2l(IA,
              JA,
              nb,
              nb,
              nb,
              nb,
              0,
              0,
              myrow,
              mycol,
              nprow,
              npcol,
              &ii,
              &jj,
              &icurrow,
              &icurcol);
  const int mp = HPL_numrocI(M, IA, nb, nb, myrow, 0, nprow);
  const int nq = HPL_numrocI(N, JA, nb, nb, mycol, 0, npcol);

  const int inxtcol = MModAdd1(icurcol, npcol);
  const int inxtrow = MModAdd1(icurrow, nprow);

  /* ptr to trailing part of A */
  PANEL->A  = A->A;
  PANEL->dA = Mptr((double*)(A->dA), ii, jj, A->ld);

  /*
   * Workspace pointers are initialized to NULL.
   */
  PANEL->L2    = nullptr;
  PANEL->dL2   = nullptr;
  PANEL->L1    = nullptr;
  PANEL->dL1   = nullptr;
  PANEL->DINFO = nullptr;
  PANEL->U     = nullptr;
  PANEL->dU    = nullptr;
  PANEL->W     = nullptr;
  PANEL->dW    = nullptr;
  PANEL->U1    = nullptr;
  PANEL->dU1   = nullptr;
  PANEL->W1    = nullptr;
  PANEL->dW1   = nullptr;
  PANEL->U2    = nullptr;
  PANEL->dU2   = nullptr;
  PANEL->W2    = nullptr;
  PANEL->dW2   = nullptr;
  // PANEL->WORK    = NULL;
  // PANEL->IWORK   = NULL;
  /*
   * Local lengths, indexes process coordinates
   */
  PANEL->nb    = nb;      /* distribution blocking factor */
  PANEL->jb    = JB;      /* panel width */
  PANEL->m     = M;       /* global # of rows of trailing part of A */
  PANEL->n     = N;       /* global # of cols of trailing part of A */
  PANEL->ia    = IA;      /* global row index of trailing part of A */
  PANEL->ja    = JA;      /* global col index of trailing part of A */
  PANEL->mp    = mp;      /* local # of rows of trailing part of A */
  PANEL->nq    = nq;      /* local # of cols of trailing part of A */
  PANEL->ii    = ii;      /* local row index of trailing part of A */
  PANEL->jj    = jj;      /* local col index of trailing part of A */
  PANEL->lda   = A->ld;   /* local leading dim of array A */
  PANEL->dlda  = A->ld;   /* local leading dim of array A */
  PANEL->prow  = icurrow; /* proc row owning 1st row of trailing A */
  PANEL->pcol  = icurcol; /* proc col owning 1st col of trailing A */
  PANEL->msgid = TAG;     /* message id to be used for panel bcast */
                          /*
                           * Initialize  ldl2 and len to temporary dummy values and Update tag for
                           * next panel
                           */
  PANEL->ldl2  = 0;       /* local leading dim of array L2 */
  PANEL->dldl2 = 0;       /* local leading dim of array L2 */
  PANEL->len   = 0;       /* length of the buffer to broadcast */
  PANEL->nu0   = 0;
  PANEL->nu1   = 0;
  PANEL->nu2   = 0;
  PANEL->ldu0  = 0;
  PANEL->ldu1  = 0;
  PANEL->ldu2  = 0;

  /*
   * Figure out the exact amount of workspace  needed by the factorization
   * and the update - Allocate that space - Finish the panel data structu-
   * re initialization.
   *
   * L1:    JB x JB in all processes
   * DINFO: 1       in all processes
   *
   * We also make an array of necessary intergers for swaps in the update.
   *
   * If nprow is 1, we just allocate an array of 2*JB integers for the swap.
   * When nprow > 1, we allocate the space for the index arrays immediate-
   * ly. The exact size of this array depends on the swapping routine that
   * will be used, so we allocate the maximum:
   *
   *       lindxU   is of size         JB +
   *       lindxA   is of size at most JB +
   *       lindxAU  is of size at most JB +
   *       permU    is of size at most JB
   *
   *       ipiv     is of size at most JB
   *
   * that is  5*JB.
   *
   * We make sure that those three arrays are contiguous in memory for the
   * later panel broadcast (using type punning to put the integer array at
   * the end.  We  also  choose  to put this amount of space right after
   * L2 (when it exist) so that one can receive a contiguous buffer.
   */

  /*Split fraction*/
  const double fraction = ALGO->frac;

  //const size_t lpiv = (5 * JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

  //const int ml2 = get_num_rows_L2(npcol, myrow, icurrow, JB, mp);

  ///* Size of LBcast message */
  ////PANEL->len = ml2 * JB + JB * JB + lpiv; // L2, L1, integer arrays

  ///* space for L */
  //const int lwork = PANEL->len + 1;

  //const int nu  = Mmax(0, (mycol == icurcol ? nq - JB : nq));
  //const int ldu = nu + JB + 256; /*extra space for potential padding*/

  ///* space for U */
  //const int uwork = JB * ldu;

  const HPL_panel_sizes psz = get_panel_sizes(A, M, N, JB, IA, JA, PANEL);
  /* Size of LBcast message */
  PANEL->len = psz.len_lbcast;
  //psz.lwork = lwork;
  //psz.uwork = uwork;
  //psz.ml2 = ml2;
  //psz.nu = nu;
  //psz.ldu = ldu;
  //psz.lpiv = lpiv;

  if(PANEL->max_lwork_size < (size_t)(psz.lwork) * sizeof(double)) {
    if(PANEL->LWORK) {
      hipFree(PANEL->dLWORK);
      free(PANEL->LWORK);
    }
    // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) +
    // (size_t)4095)/(size_t)4096)*(size_t)4096;
    const size_t numbytes = (size_t)(psz.lwork) * sizeof(double);

    if(deviceMalloc((void**)&(PANEL->dLWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Device memory allocation failed for L workspace.");
    }
    if(hostMalloc((void**)&(PANEL->LWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for L workspace.");
    }

    PANEL->max_lwork_size = (size_t)(psz.lwork) * sizeof(double);
  }
  if(PANEL->max_uwork_size < (size_t)(psz.uwork) * sizeof(double)) {
    if(PANEL->UWORK) {
      hipFree(PANEL->dUWORK);
      free(PANEL->UWORK);
    }
    // size_t numbytes = (((size_t)((size_t)(uwork) * sizeof( double )) +
    // (size_t)4095)/(size_t)4096)*(size_t)4096;
    size_t numbytes = (size_t)(psz.uwork) * sizeof(double);

    if(deviceMalloc((void**)&(PANEL->dUWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Device memory allocation failed for U workspace.");
    }
    if(hostMalloc((void**)&(PANEL->UWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for U workspace.");
    }

    PANEL->max_uwork_size = (size_t)(psz.uwork) * sizeof(double);
  }

  // Initialize the pointers of the panel structure
  initialize_panel_pointers(A, JB, icurrow, icurcol, mp, psz.ml2, psz.nu, psz.lpiv, fraction, PANEL);

  *(PANEL->DINFO) = 0.0;

  //const int liwork = get_index_workspace_len(nprow, JB, mp);
  //psz.l_i_work = (size_t)get_index_workspace_len(nprow, JB, mp);

  if(PANEL->max_iwork_size < psz.l_i_work * sizeof(int)) {
    if(PANEL->IWORK) {
        free(PANEL->IWORK);
    }
    const size_t numbytes = psz.l_i_work * sizeof(int);

    if(HPL_malloc((void**)&(PANEL->IWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for integer workspace.");
    }
    PANEL->max_iwork_size = psz.l_i_work * sizeof(int);
  }

  if(psz.l_i_work) {
    *(PANEL->IWORK) = -1;
  }

  /*Finally, we need 4 + 4*JB entries of scratch for pdfact */
  //const size_t lfwork = (size_t)(((4 + ((unsigned int)(JB) << 1)) << 1));
  //psz.l_f_work = (size_t)(((4 + ((unsigned int)(JB) << 1)) << 1));
  if(PANEL->max_fwork_size < psz.l_f_work * sizeof(double)) {
    if(PANEL->fWORK) {
        free(PANEL->fWORK);
    }
    const size_t numbytes = psz.l_f_work * sizeof(double);

    if(HPL_malloc((void**)&(PANEL->fWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for pdfact scratch workspace.");
    }
    PANEL->max_fwork_size = psz.l_f_work * sizeof(double);
  }
}
