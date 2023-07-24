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
#ifndef HPL_PFACT_HPP
#define HPL_PFACT_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"

#include "hpl_pgesv.hpp"
#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef void (*HPL_T_PFA_FUN)(HPL_T_panel*,
                              const int,
                              const int,
                              const int,
                              double*,
                              int,
                              int,
                              double*,
                              int*,
                              HPL_Comm_impl_type);

typedef void (*HPL_T_RFA_FUN)(HPL_T_panel*,
                              const int,
                              const int,
                              const int,
                              double*,
                              int,
                              int,
                              double*,
                              int*,
                              HPL_Comm_impl_type);
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_dlocmax(HPL_T_panel*,
                 const int,
                 const int,
                 const int,
                 double*,
                 int,
                 int,
                 int*,
                 double*);

void HPL_dlocswpN(HPL_T_panel*, const int, const int, double*);
void HPL_dlocswpT(HPL_T_panel*, const int, const int, double*);
void HPL_pdmxswp(HPL_T_panel*, const int, const int, const int, double*, HPL_Comm_impl_type);

void HPL_pdpancrN(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdpancrT(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdpanllN(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdpanllT(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdpanrlN(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdpanrlT(HPL_T_panel*,
                  const int,
                  const int,
                  const int,
                  double*,
                  int,
                  int,
                  double*,
                  int*,
                  HPL_Comm_impl_type);

void HPL_pdrpancrN(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdrpancrT(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdrpanllN(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdrpanllT(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdrpanrlN(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdrpanrlT(HPL_T_panel*,
                   const int,
                   const int,
                   const int,
                   double*,
                   int,
                   int,
                   double*,
                   int*,
                   HPL_Comm_impl_type);

void HPL_pdfact(HPL_T_panel*, HPL_Comm_impl_type);

#endif
/*
 * End of hpl_pfact.hpp
 */
