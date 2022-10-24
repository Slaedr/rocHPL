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

void HPL_pdupdateNT(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdupdateNT applies the row interchanges and updates part of the
   * trailing  (using the panel PANEL) submatrix.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel (to be updated) information.
   *
   * ---------------------------------------------------------------------
   */

  double *Aptr, *L1ptr, *L2ptr, *Uptr, *dpiv;
  int*    dipiv;

  int curr, i, jb, lda, ldl2, LDU, mp, n, nb;

  /* ..
   * .. Executable Statements ..
   */
  nb   = PANEL->nb;
  jb   = PANEL->jb;
  n    = PANEL->nq;
  lda  = PANEL->dlda;
  Aptr = PANEL->dA;

  if(UPD == HPL_LOOK_AHEAD) {
    Uptr = PANEL->dU;
    LDU  = PANEL->ldu0;
    n    = Mmin(PANEL->nu0, n);
  } else if(UPD == HPL_UPD_1) {
    Uptr = PANEL->dU1;
    LDU  = PANEL->ldu1;
    n    = Mmin(PANEL->nu1, n);
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    Aptr = Mptr(Aptr, 0, PANEL->nu0, lda);
  } else if(UPD == HPL_UPD_2) {
    Uptr = PANEL->dU2;
    LDU  = PANEL->ldu2;
    n    = Mmin(PANEL->nu2, n);
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    Aptr = Mptr(Aptr, 0, PANEL->nu0 + PANEL->nu1, lda);
  }

  /*
   * There is nothing to update, enforce the panel broadcast.
   */
  if((n <= 0) || (jb <= 0)) { return; }

  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  curr  = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
  L2ptr = PANEL->dL2;
  L1ptr = PANEL->dL1;
  ldl2  = PANEL->dldl2;
  mp    = PANEL->mp - (curr != 0 ? jb : 0);

  const double one  = 1.0;
  const double mone = -1.0;

  /*
   * Update
   */
  if(PANEL->grid->nprow == 1) {
    /*
     * 1 x Q case
     */
    cublasDtrsm(handle,
                  CUBLAS_SIDE_LEFT,
                  CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_N,
                  CUBLAS_DIAG_UNIT,
                  jb,
                  n,
                  &one,
                  L1ptr,
                  jb,
                  Aptr,
                  lda);

    HPL_dlatcpy_gpu(n, jb, Aptr, lda, Uptr, LDU);
  } else {
    /*
     * Compute redundantly row block of U and update trailing submatrix
     */
    cublasDtrsm(handle,
                  CUBLAS_SIDE_RIGHT,
                  CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_T,
                  CUBLAS_DIAG_UNIT,
                  n,
                  jb,
                  &one,
                  L1ptr,
                  jb,
                  Uptr,
                  LDU);
  }

  /*
   * Queue finishing the update
   */
  if(curr != 0) {
    cudaEventRecord(dgemmStart[UPD], stream);
    cublasDgemm(handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_T,
                  mp,
                  n,
                  jb,
                  &mone,
                  L2ptr,
                  ldl2,
                  Uptr,
                  LDU,
                  &one,
                  Mptr(Aptr, jb, 0, lda),
                  lda);
    cudaEventRecord(dgemmStop[UPD], stream);

    if(PANEL->grid->nprow > 1) HPL_dlatcpy_gpu(jb, n, Uptr, LDU, Aptr, lda);
  } else {
    cudaEventRecord(dgemmStart[UPD], stream);
    cublasDgemm(handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_T,
                  mp,
                  n,
                  jb,
                  &mone,
                  L2ptr,
                  ldl2,
                  Uptr,
                  LDU,
                  &one,
                  Aptr,
                  lda);
    cudaEventRecord(dgemmStop[UPD], stream);
  }

  cudaEventRecord(update[UPD], stream);
}
