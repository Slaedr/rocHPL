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
#ifndef HPL_PMATGEN_HPP
#define HPL_PMATGEN_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_pgesv.hpp"
#include "hpl_ptest.hpp"

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_MULT 6364136223846793005UL
#define HPL_IADD 1UL
#define HPL_DIVFAC 2147483648.0
#define HPL_POW16 65536.0
#define HPL_HALF 0.5

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_xjumpm(const int      JUMPM,
                const uint64_t MULT,
                const uint64_t IADD,
                const uint64_t IRANN,
                uint64_t&      IRANM,
                uint64_t&      IAM,
                uint64_t&      ICM);

void HPL_pdrandmat(const HPL_T_grid*,
                   const int,
                   const int,
                   const int,
                   double*,
                   const int,
                   const int);

int HPL_pdmatgen(HPL_T_test*,
                 const HPL_T_grid*,
                 const HPL_T_palg*,
                 HPL_T_pmat*,
                 const int,
                 const int);

/**
 * Read a matrix from files.
 */
void HPL_pdreadmat(const HPL_T_grid* grid,
                   int nrows_global, int ncols_global,
                   std::string path_prefix,
                   ornl_hpl::matrix_dir_type mdtype,
                   HPL_T_pmat* mat);

template <typename scalar>
void device_copy_2d_block(size_t source_ld, int nrows, int ncols, const scalar *source,
                          size_t dest_ld, scalar *const dest);

void HPL_gather_solution(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
                         double *const hX);

void HPL_gather_write_solution(const HPL_T_grid *grid, const HPL_T_pmat *mat,
                               const std::string& matrix_dir);

void HPL_write_solution_by_blocks(const HPL_T_grid *grid, const HPL_T_pmat *mat,
                                  const std::string& matrix_dir);

void HPL_pdmatfree(HPL_T_pmat*);

#endif
/*
 * End of hpl_pmatgen.hpp
 */
