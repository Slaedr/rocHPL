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
#ifndef HPL_PGESV_HPP
#define HPL_PGESV_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"
#include "hpl_comm.hpp"
#include "hpl_pmat.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pfact.hpp"

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define MSGID_BEGIN_PFACT 1001 /* message id ranges */
#define MSGID_END_PFACT 2000
#define MSGID_BEGIN_FACT 2001
#define MSGID_END_FACT 3000
#define MSGID_BEGIN_PTRSV 3001
#define MSGID_END_PTRSV 4000

#define MSGID_BEGIN_COLL 9001
#define MSGID_END_COLL 10000
/*
 * ---------------------------------------------------------------------
 * #define macros definitions
 * ---------------------------------------------------------------------
 */
#define MNxtMgid(id_, beg_, end_) (((id_) + 1 > (end_) ? (beg_) : (id_) + 1))
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */

void HPL_pipid(HPL_T_panel*, int*, int*);
void HPL_piplen(HPL_T_panel*, const int, const int*, int*, int*);
void HPL_perm(const int, int*, int*, int*);

void HPL_plindx(HPL_T_panel*,
                const int,
                const int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*);

void HPL_pdlaswp_start(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdlaswp_exchange(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdlaswp_end(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdupdateNT(HPL_T_panel*, const HPL_T_UPD);
void HPL_pdupdateTT(HPL_T_panel*, const HPL_T_UPD);
void HPL_pdgesv(HPL_T_grid*, HPL_T_palg*, HPL_T_pmat*);
void HPL_pdtrsv(HPL_T_grid*, HPL_T_pmat*);

#endif
/*
 * End of hpl_pgesv.hpp
 */
