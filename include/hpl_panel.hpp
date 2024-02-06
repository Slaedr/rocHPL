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
#ifndef HPL_PANEL_HPP
#define HPL_PANEL_HPP

/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"
#include "hpl_pgesv_types.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
struct HPL_T_panel {
  const HPL_T_grid* grid;   /* ptr to the process grid */
  const HPL_T_palg* algo;   /* ptr to the algo parameters */
  HPL_T_pmat* pmat;   /* ptr to the local array info */
  double*            A;      /* ptr to trailing part of A */
  double*            dA;     /* ptr to trailing part of A */
  double*            LWORK;  /* L work space */
  double*            dLWORK; /* L device-copy work space */
  double*            UWORK;  /* U work space */
  double*            dUWORK; /* U device-copy work space */
  double*            fWORK;  /* pdfact work space */
  double*            L2;     /* ptr to L */
  double*            L1;     /* ptr to jb x jb upper block of A */
  double*            dL2;    /* ptr to L */
  double*            dL1;    /* ptr to jb x jb upper block of A */
  double*            DINFO;  /* ptr to replicated scalar info */
  double*            dDINFO; /* ptr to replicated scalar info */
  int*               ipiv;
  int*               dipiv;
  int*               lindxA;
  int*               dlindxA;
  int*               lindxAU;
  int*               dlindxAU;
  int*               lindxU;
  int*               dlindxU;
  int*               permU;
  int*               dpermU;
  double*            U;   /* ptr to U */
  double*            dU;  /* ptr to U */
  double*            W;   /* ptr to W */
  double*            dW;  /* ptr to W */
  double*            U1;  /* ptr to U1 */
  double*            dU1; /* ptr to U1 */
  double*            W1;  /* ptr to W1 */
  double*            dW1; /* ptr to W1 */
  double*            U2;  /* ptr to U2 */
  double*            dU2; /* ptr to U2 */
  double*            W2;  /* ptr to W2 */
  double*            dW2; /* ptr to W2 */
  int                nu0;
  int                nu1;
  int                nu2;
  int                ldu0;
  int                ldu1;
  int                ldu2;
  int*               IWORK;      /* integer workspace for swapping */
  void*              buffers[2]; /* buffers for panel bcast */
  int                counts[2];  /* counts for panel bcast */
  MPI_Datatype       dtypes[2];  /* data types for panel bcast */
  MPI_Request        request[1]; /* requests for panel bcast */
  MPI_Status         status[1];  /* status for panel bcast */
  int                nb;         /* distribution blocking factor */
  int                jb;         /* panel width */
  int                m;          /* global # of rows of trailing part of A */
  int                n;          /* global # of cols of trailing part of A */
  int                ia;         /* global row index of trailing part of A */
  int                ja;         /* global col index of trailing part of A */
  int                mp;         /* local # of rows of trailing part of A */
  int                nq;         /* local # of cols of trailing part of A */
  int                ii;         /* local row index of trailing part of A */
  int                jj;         /* local col index of trailing part of A */
  int                lda;        /* local leading dim of array A */
  int                dlda;       /* local leading dim of array A */
  int                prow;       /* proc. row owning 1st row of trail. A */
  int                pcol;       /* proc. col owning 1st col of trail. A */
  int                msgid;      /* message id for panel bcast */
  int                ldl2;       /* local leading dim of array L2 */
  int                dldl2;      /* local leading dim of array L2 */
  int                len;        /* length of the buffer to broadcast */
  // TODO: Change these to size_t to avoid type mismatch (see HPL_pdpanel_init)
  size_t       max_pinned_work_size; /* largest size of pinned A space */
  size_t       max_lwork_size;       /* largest size of WORK space */
  size_t       max_uwork_size;       /* largest size of WORK space */
  size_t       max_iwork_size;       /* largest size of IWORK space */
  size_t       max_fwork_size;       /* largest size of fWORK space */
  size_t       free_work_now;        /* should we deallocate */
};

/*
 * ---------------------------------------------------------------------
 * panel function prototypes
 * ---------------------------------------------------------------------
 */

void HPL_pdpanel_new(HPL_T_grid*,
                     HPL_T_palg*,
                     const int,
                     const int,
                     const int,
                     HPL_T_pmat*,
                     const int,
                     const int,
                     const int,
                     HPL_T_panel**);

void HPL_pdpanel_init(HPL_T_grid*,
                      HPL_T_palg*,
                      const int,
                      const int,
                      const int,
                      HPL_T_pmat*,
                      const int,
                      const int,
                      const int,
                      HPL_T_panel*);

int  HPL_pdpanel_disp(HPL_T_panel**);
int  HPL_pdpanel_free(HPL_T_panel*);
void HPL_pdpanel_SendToHost(HPL_T_panel*);
void HPL_pdpanel_SendToDevice(HPL_T_panel*);
void HPL_pdpanel_Wait(HPL_T_panel* PANEL);
int  HPL_pdpanel_bcast(HPL_T_panel*);

// Helper functions

/// Get number or rows in the L2 factor matrix
int get_num_rows_L2(int npcol, int myrow, int current_prow,
                    int panel_ncols, int local_nrows);

/** \brief Get length of workspace for storing indices.
 *
 * If nprow is 1, we just allocate an array of JB (panel_ncols) integers to store the
 * pivot IDs during factoring, and a scratch array of mp integers.
 * When nprow > 1, we allocate the space for the index arrays immediate-
 * ly. The exact size of this array depends on the swapping routine that
 * will be used, so we allocate the maximum:
 *
 *    IWORK[0] is of size at most 1      +
 *    IPL      is of size at most 1      +
 *    IPID     is of size at most 4 * JB +
 *    IPIV     is of size at most JB     +
 *    SCRATCH  is of size at most MP
 *
 *    ipA      is of size at most 1      +
 *    iplen    is of size at most NPROW  + 1 +
 *    ipcounts is of size at most NPROW  +
 *    ioffsets is of size at most NPROW  +
 *    iwork    is of size at most MAX( 2*JB, NPROW+1 ).
 *
 * that is  mp + 4 + 5*JB + 3*NPROW + MAX( 2*JB, NPROW+1 ).
 *
 * We use the fist entry of this to work array  to indicate  whether the
 * the  local  index arrays have already been computed,  and if yes,  by
 * which function:
 *    IWORK[0] = -1: no index arrays have been computed so far;
 *    IWORK[0] =  1: HPL_pdlaswp already computed those arrays;
 * This allows to save some redundant and useless computations.
 */
int get_index_workspace_len(int nprow, int panel_ncols, int local_nrows);

#endif
/*
 * End of hpl_panel.hpp
 */
