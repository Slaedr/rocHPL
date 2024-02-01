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
#include "hpl_pgesv_types.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"
#include "hpl_comm.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pfact.hpp"

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
