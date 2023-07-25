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

#include <vector>

/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pgesv.hpp"

#include "hpl_ptimer.hpp"
#include "hpl_pmatgen.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
typedef struct HPL_S_test {
  double epsil; /* epsilon machine */
  double thrsh; /* threshold */
  FILE*  outfp; /* output stream (only in proc 0) */
  int    kfail; /* # of tests failed */
  int    kpass; /* # of tests passed */
  int    kskip; /* # of tests skipped */
  int    ktest; /* total number of tests */
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
#define HPL_TIMING_N 8       /* number of timers defined below */
#define HPL_TIMING_RPFACT 11 /* starting from here, contiguous */
#define HPL_TIMING_PFACT 12
#define HPL_TIMING_MXSWP 13
#define HPL_TIMING_COPY 14
#define HPL_TIMING_LBCAST 15
#define HPL_TIMING_LASWP 16
#define HPL_TIMING_UPDATE 17
#define HPL_TIMING_PTRSV 18

struct HPL_Test_params {
    /// Matrix sizes to run, N
    std::vector<int> matrix_sizes;
    /// Block sizes to test, NB
    std::vector<int> bs;
    /** Specifies the process mapping onto the no-
     * des of the  MPI machine configuration.  PMAPPIN  defaults  to
     * row-major ordering.
     * AKA: PMAPPIN
     */
    HPL_T_ORDER process_ordering;
    /// Number of process-row and process-column settings to test.
    //int npqs;
    /// Global process-rows counts to test, AKA P
    std::vector<int> gl_proc_rows;
    /// Global process-columns counts to test, AKA Q
    std::vector<int> gl_proc_cols;
    /// Node-local process rows, p
    int loc_proc_rows;
    /// Node-local process columns, q
    int loc_proc_cols;
    /// Panel factorizations to run, PF
    std::vector<HPL_T_FACT> panel_facts;
    /// Recursive stopping criteria, NBM
    std::vector<int> recursive_stop_crit;
    /// Numbers of panels in recursion, NDV
    std::vector<int> num_panels_recursion;
    /// Recursive factorization algorithms, RF
    std::vector<HPL_T_FACT> recursive_facts;
    /** Broadcast algorithms, TP
     * This is only used if \ref bcast_type is HPL_COMM_CUSTOM_IMPL.
     */
    std::vector<HPL_T_TOP> bcast_algos;
    /// Lookahead depths, DH
    std::vector<int> lookahead_depths;
    /// Swapping algo to be used on all tests
    HPL_T_SWAP fswap;
    /// Swapping threshold as a number of columns when mixed swapping algo is chosen, TSWAP
    int swap_threshold_cols;
    /** Specifies whether the upper triangle of the panels of columns should be stored
     * in non-transposed form (1) or transposed form (0). AKA L1NOTRAN.
     */
    bool L1_no_transpose;
    /** Specifies whether the panels of rows should be stored
     * in non-transposed form (1) or transposed form (0). AKA UNOTRAN.
     */
    bool U_no_transpose;
    /// Specifies whether equilibration during swap-broadcast of panel of rows should be performed.
    bool equil;
    /// Memory alignment of dynamically allocated buffers in double-precision words, ALIGN
    int mem_align;
    /// Percentage in which to split the trailing update.
    double frac;
    /// Type of implementation to use for broadcast
    HPL_Comm_impl_type bcast_type;
    /// Type of implementation to use for all-reduce in panel factorization
    HPL_Comm_impl_type allreduce_dmxswp_type;
    /// Type of implementation to use for allgatherv
    HPL_Comm_impl_type allgatherv_type;
    /// Type of implementation to use for scatterv
    HPL_Comm_impl_type scatterv_type;
};

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
HPL_Test_params HPL_pdinfo(int    ARGC,
                char** ARGV,
                HPL_T_test*);

void HPL_pdtest(HPL_T_test*, HPL_T_grid*, HPL_T_palg*, const int, const int);
void HPL_InitGPU(const HPL_T_grid* GRID);
void HPL_FreeGPU();

#endif
/*
 * End of hpl_ptest.hpp
 */
