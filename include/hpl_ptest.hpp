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
#ifndef HPL_PTEST_HPP
#define HPL_PTEST_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include <string>

#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pgesv.hpp"
#include "hpl_pmat.hpp"

#include "hpl_ptimer.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */

namespace ornl_hpl {

/**
 * Type of directory structure from which matrix blocks are read,
 * if applicable.
 *
 * \sa HPL_pdreadmat
 */
enum class matrix_dir_type {
    /**
     * All matrix blocks are in one directory, named as A_i_j.npy where
     * i and j are row-block and column-block indices.
     */
    flat,
    /**
     * Matrix blocks belonging to row block i are stored in a subdirectory
     * named row_i (where i is written always using 5 characters).
     */
    row_block_dirs
};

}

typedef struct HPL_S_test {
  double epsil; /* epsilon machine */
  double thrsh; /* threshold */
  FILE*  outfp; /* output stream (only in proc 0) */
  int    kfail; /* # of tests failed */
  int    kpass; /* # of tests passed */
  int    kskip; /* # of tests skipped */
  int    ktest; /* total number of tests */
  /*
   * Number of times to split the input block size.
   */
  int    refine_blocks;
  std::string matrix_dir;  //< Directory containing matrix files
  ornl_hpl::matrix_dir_type mdtype; //< Type of matrix directory structure
} HPL_T_test;

/*
 * ---------------------------------------------------------------------
 * #define macro constants for testing only
 * ---------------------------------------------------------------------
 */
#define HPL_LINE_MAX 256
#define HPL_MAX_PARAM 20
#define HPL_ISEED 100
/*
 * ---------------------------------------------------------------------
 * global timers for timing analysis only
 * ---------------------------------------------------------------------
 */
#define HPL_TIMING_BEG 11    /* timer 0 reserved, used by main */
#define HPL_TIMING_N 13      /* number of timers defined below */
#define HPL_TIMING_FACTOR_N 8 /* number of timers for factorization */
#define HPL_TIMING_IO_N 4
#define HPL_TIMING_RPFACT 11 /* starting from here, contiguous */
#define HPL_TIMING_PFACT 12
#define HPL_TIMING_MXSWP 13
#define HPL_TIMING_COPY 14
#define HPL_TIMING_LBCAST 15
#define HPL_TIMING_LASWP 16
#define HPL_TIMING_UPDATE 17
#define HPL_TIMING_PTRSV 18
#define HPL_TIMING_IO 19
#define HPL_TIMING_IO_MAT 20
#define HPL_TIMING_IO_VEC_READ 21
#define HPL_TIMING_IO_VEC_WRITE 22
#define HPL_TIMING_MAT_VEC_REDISTRIBUTE 23
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_pdinfo(int    ARGC,
                char** ARGV,
                HPL_T_test*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_ORDER*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_FACT*,
                int*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_FACT*,
                int*,
                HPL_T_TOP*,
                int*,
                int*,
                HPL_T_SWAP*,
                int*,
                int*,
                int*,
                int*,
                int*,
                double*);

void HPL_pdtest(HPL_T_test*, HPL_T_grid*, HPL_T_palg*, const int, const int);
void HPL_InitGPU(const HPL_T_grid* GRID);
void HPL_FreeGPU();

#endif
/*
 * End of hpl_ptest.hpp
 */
