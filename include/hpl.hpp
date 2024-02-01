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
#ifndef HPL_HPP
#define HPL_HPP
/*
 * ---------------------------------------------------------------------
 * HPL default compile options that can overridden in the cmake
 * ---------------------------------------------------------------------
 */
#ifndef HPL_DETAILED_TIMING /* Do not enable detailed timings */
#define HPL_NO_DETAILED_TIMING
#endif

#undef HPL_USE_COLLECTIVES
//#define HPL_USE_COLLECTIVES

#include "hpl_version.hpp"
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pfact.hpp"
#include "hpl_pgesv.hpp"

#include "hpl_ptimer.hpp"
#include "hpl_pmatgen.hpp"
#include "hpl_ptest.hpp"

#endif
/*
 * End of hpl.hpp
 */
