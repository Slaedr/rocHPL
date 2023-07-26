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
#include <iostream>
#include <cstdio>
#include <cstring>

HPL_Test_params HPL_pdinfo(int          ARGC,
                char**       ARGV,
                HPL_T_test*  TEST)
{
                
    int nS{-1}, N[HPL_MAX_PARAM], nBS{-1}, NB[HPL_MAX_PARAM];
    int nPQS{-1}, P[HPL_MAX_PARAM], Q[HPL_MAX_PARAM], nPFS{-1};
    HPL_T_FACT  PF[HPL_MAX_PARAM];
    int         nBMS{-1}, NBM[HPL_MAX_PARAM], nDVS{-1}, NDV[HPL_MAX_PARAM], nRFS{-1};
    HPL_T_FACT  RF[HPL_MAX_PARAM];
    int         nTPS{-1};
    HPL_T_TOP   TP[HPL_MAX_PARAM];
    int         nDHS{-1}, DH[HPL_MAX_PARAM];
    int         L1NOTRAN{-1}, UNOTRAN{-1}, EQUIL{-1};
  /*
   * Purpose
   * =======
   *
   * HPL_pdinfo reads  the  startup  information for the various tests and
   * transmits it to all processes.
   *
   * Arguments
   * =========
   *
   * TEST    (global output)               HPL_T_test *
   *         On entry, TEST  points to a testing data structure.  On exit,
   *         the fields of this data structure are initialized as follows:
   *         TEST->outfp  specifies the output file where the results will
   *         be printed.  It is only defined and used by  the process 0 of
   *         the grid.  TEST->thrsh specifies the threshhold value for the
   *         test ratio.  TEST->epsil is the relative machine precision of
   *         the distributed computer.  Finally  the test counters, kfail,
   *         kpass, kskip, ktest are initialized to zero.
   *
   * NS      (global output)               int *
   *         On exit,  NS  specifies the number of different problem sizes
   *         to be tested. NS is less than or equal to HPL_MAX_PARAM.
   *
   * N       (global output)               int *
   *         On entry, N is an array of dimension HPL_MAX_PARAM.  On exit,
   *         the first NS entries of this array contain the  problem sizes
   *         to run the code with.
   *
   * nBS     (global output)               int *
   *         On exit,  nBS  specifies the number of different distribution
   *         blocking factors to be tested. nBS must be less than or equal
   *         to HPL_MAX_PARAM.
   *
   * NB      (global output)               int *
   *         On exit,  PMAPPIN  specifies the process mapping onto the no-
   *         des of the  MPI machine configuration.  PMAPPIN  defaults  to
   *         row-major ordering.
   *
   * PMAPPIN (global output)               HPL_T_ORDER *
   *         On entry, NB is an array of dimension HPL_MAX_PARAM. On exit,
   *         the first nBS entries of this array contain the values of the
   *         various distribution blocking factors, to run the code with.
   *
   * NPQS    (global output)               int *
   *         On exit, NPQS  specifies the  number of different values that
   *         can be used for P and Q, i.e., the number of process grids to
   *         run  the  code with.  NPQS must be  less  than  or  equal  to
   *         HPL_MAX_PARAM.
   *
   * P       (global output)               int *
   *         On entry, P  is an array of dimension HPL_MAX_PARAM. On exit,
   *         the first NPQS entries of this array contain the values of P,
   *         the number of process rows of the  NPQS grids to run the code
   *         with.
   *
   * Q       (global output)               int *
   *         On entry, Q  is an array of dimension HPL_MAX_PARAM. On exit,
   *         the first NPQS entries of this array contain the values of Q,
   *         the number of process columns of the  NPQS  grids to  run the
   *         code with.
   *
   * p       (global output)               int *
   *         On exit, p specifies the number of rows in the node-local MPI
   *         grid
   *
   * q       (global output)               int *
   *         On exit, q specifies the number of columns in the node-local
   *         MPI grid
   *
   * NPFS    (global output)               int *
   *         On exit, NPFS  specifies the  number of different values that
   *         can be used for PF : the panel factorization algorithm to run
   *         the code with. NPFS is less than or equal to HPL_MAX_PARAM.
   *
   * PF      (global output)               HPL_T_FACT *
   *         On entry, PF is an array of dimension HPL_MAX_PARAM. On exit,
   *         the first  NPFS  entries  of this array  contain  the various
   *         panel factorization algorithms to run the code with.
   *
   * NBMS    (global output)               int *
   *         On exit,  NBMS  specifies  the  number  of  various recursive
   *         stopping criteria  to be tested.  NBMS  must be  less than or
   *         equal to HPL_MAX_PARAM.
   *
   * NBM     (global output)               int *
   *         On entry,  NBM  is an array of  dimension  HPL_MAX_PARAM.  On
   *         exit, the first NBMS entries of this array contain the values
   *         of the various recursive stopping criteria to be tested.
   *
   * NDVS    (global output)               int *
   *         On exit,  NDVS  specifies  the number  of various numbers  of
   *         panels in recursion to be tested.  NDVS is less than or equal
   *         to HPL_MAX_PARAM.
   *
   * NDV     (global output)               int *
   *         On entry,  NDV  is an array of  dimension  HPL_MAX_PARAM.  On
   *         exit, the first NDVS entries of this array contain the values
   *         of the various numbers of panels in recursion to be tested.
   *
   * NRFS    (global output)               int *
   *         On exit, NRFS  specifies the  number of different values that
   *         can be used for RF : the recursive factorization algorithm to
   *         be tested. NRFS is less than or equal to HPL_MAX_PARAM.
   *
   * RF      (global output)               HPL_T_FACT *
   *         On entry, RF is an array of dimension HPL_MAX_PARAM. On exit,
   *         the first  NRFS  entries  of  this array contain  the various
   *         recursive factorization algorithms to run the code with.
   *
   * NTPS    (global output)               int *
   *         On exit, NTPS  specifies the  number of different values that
   *         can be used for the  broadcast topologies  to be tested. NTPS
   *         is less than or equal to HPL_MAX_PARAM.
   *
   * TP      (global output)               HPL_T_TOP *
   *         On entry, TP is an array of dimension HPL_MAX_PARAM. On exit,
   *         the  first NTPS  entries of this  array  contain  the various
   *         broadcast (along rows) topologies to run the code with.
   *
   * NDHS    (global output)               int *
   *         On exit, NDHS  specifies the  number of different values that
   *         can be used for the  lookahead depths to be  tested.  NDHS is
   *         less than or equal to HPL_MAX_PARAM.
   *
   * DH      (global output)               int *
   *         On entry,  DH  is  an array of  dimension  HPL_MAX_PARAM.  On
   *         exit, the first NDHS entries of this array contain the values
   *         of lookahead depths to run the code with.  Such a value is at
   *         least 0 (no-lookahead) or greater than zero.
   *
   * FSWAP   (global output)               HPL_T_SWAP *
   *         On exit, FSWAP specifies the swapping algorithm to be used in
   *         all tests.
   *
   * TSWAP   (global output)               int *
   *         On exit,  TSWAP  specifies the swapping threshold as a number
   *         of columns when the mixed swapping algorithm was chosen.
   *
   * L1NOTRA (global output)               int *
   *         On exit, L1NOTRAN specifies whether the upper triangle of the
   *         panels of columns  should  be stored  in  no-transposed  form
   *         (L1NOTRAN=1) or in transposed form (L1NOTRAN=0).
   *
   * UNOTRAN (global output)               int *
   *         On exit, UNOTRAN  specifies whether the panels of rows should
   *         be stored in  no-transposed form  (UNOTRAN=1)  or  transposed
   *         form (UNOTRAN=0) during their broadcast.
   *
   * EQUIL   (global output)               int *
   *         On exit,  EQUIL  specifies  whether  equilibration during the
   *         swap-broadcast  of  the  panel of rows  should  be  performed
   *         (EQUIL=1) or not (EQUIL=0).
   *
   * ALIGN   (global output)               int *
   *         On exit,  ALIGN  specifies the alignment  of  the dynamically
   *         allocated buffers in double precision words. ALIGN is greater
   *         than zero.
   *
   * FRAC    (global output)               double *
   *         On exit,  FRAC  specifies the percentage in which to split the
   *         the trailing update.
   *
   * ---------------------------------------------------------------------
   */

  char file[HPL_LINE_MAX], line[HPL_LINE_MAX], auth[HPL_LINE_MAX],
      num[HPL_LINE_MAX];
  FILE* infp;
  int*  iwork = NULL;
  char* lineptr;
  int   error = 0, fid, lwork, maxp, nprocs, rank, size;
  HPL_Test_params params;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /*
   * Initialize the TEST data structure with default values
   */
  TEST->outfp = stderr;
  TEST->epsil = 2.0e-16;
  TEST->thrsh = 16.0;
  TEST->kfail = TEST->kpass = TEST->kskip = TEST->ktest = 0;

  // parse settings
  int         _P = 1, _Q = 1, n = 45312, nb = 384;
  int         _p = -1, _q = -1;
  bool        cmdlinerun    = false;
  bool        inputfile     = false;
  double      frac          = 0.6;
  std::string inputFileName = "HPL.dat";
  std::string outfile_path{"HPL.out"};
  bool use_mpi_lbcast = false, use_mpi_allreduce_dmxswp = false, use_mpi_allgatherv = false,
       use_mpi_scatterv = false;

  for(int i = 1; i < ARGC; i++) {
    if(strcmp(ARGV[i], "-h") == 0 || strcmp(ARGV[i], "--help") == 0) {
      if(rank == 0) {
        std::cout
            << "rocHPL client command line options:                      "
               "           \n"
               "-P  [ --ranksP ] arg (=1)          Specific MPI grid "
               "size: the number of      \n"
               "                                   rows in MPI grid.     "
               "                     \n"
               "-Q  [ --ranksQ ] arg (=1)          Specific MPI grid "
               "size: the number of      \n"
               "                                   columns in MPI grid.  "
               "                     \n"
               "-N  [ --sizeN ]  arg (=45312)      Specific matrix size: "
               "the number of rows   \n"
               "                                   /columns in global "
               "matrix.                 \n"
               "-NB [ --sizeNB ] arg (=384)        Specific panel size: "
               "the number of rows    \n"
               "                                   /columns in panels.   "
               "                     \n"
               "-f  [ --frac ] arg (=0.6)          Specific update split: "
               "the percentage to    \n"
               "                                   split the trailing "
               "submatrix.           \n"
               "-i  [ --input ]  arg (=HPL.dat)    Input file. When set, "
               "all other commnand   \n"
               "                                   line parameters are "
               "ignored, and problem   \n"
               "                                   parameters are read "
               "from input file.       \n"
               "-h  [ --help ]                     Produces this help "
               "message                 \n"
               "--version                          Prints the version "
               "number                  \n"
               "-o [ --output_file ]               Name of output file [HPL.out]\n"
               "--use_mpi_lbcast                   Use MPI collective for panel broadcast.\n"
               "--use_mpi_allreduce_dmxswp         Use MPI collective for allreduce in pivoting.\n"
               "--use_mpi_scatterv                 Use MPI collective for scatter during row swaps.\n"
               "--use_mpi_allgatherv               Use MPI collective for all-gather during row swaps.\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);
    }

    if(strcmp(ARGV[i], "--version") == 0) {
      if(rank == 0) {
        std::cout << "rocHPL version: " << __ROCHPL_VER_MAJOR << "."
                  << __ROCHPL_VER_MINOR << "." << __ROCHPL_VER_PATCH
                  << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);
    }

    if(strcmp(ARGV[i], "-P") == 0 || strcmp(ARGV[i], "--ranksP") == 0) {
      _P         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(_P < 1) {
        if(rank == 0)
          HPL_pwarn(stderr,
                    __LINE__,
                    "HPL_pdinfo",
                    "Illegal value for P. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-Q") == 0 || strcmp(ARGV[i], "--ranksQ") == 0) {
      _Q         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(_Q < 1) {
        if(rank == 0)
          HPL_pwarn(stderr,
                    __LINE__,
                    "HPL_pdinfo",
                    "Illegal value for Q. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-p") == 0) {
      _p         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
    }
    if(strcmp(ARGV[i], "-q") == 0) {
      _q         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
    }

    if(strcmp(ARGV[i], "-N") == 0 || strcmp(ARGV[i], "--sizeN") == 0) {
      n          = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(n < 1) {
        if(rank == 0)
          HPL_pwarn(stderr,
                    __LINE__,
                    "HPL_pdinfo",
                    "Illegal value for N. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-NB") == 0 || strcmp(ARGV[i], "--sizeNB") == 0) {
      nb         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(nb < 1) {
        if(rank == 0)
          HPL_pwarn(stderr,
                    __LINE__,
                    "HPL_pdinfo",
                    "Illegal value for NB. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-f") == 0 || strcmp(ARGV[i], "--frac") == 0) {
      frac = atof(ARGV[i + 1]);
      i++;
    }
    if(strcmp(ARGV[i], "-i") == 0 || strcmp(ARGV[i], "--input") == 0) {
      inputFileName = ARGV[i + 1];
      inputfile     = true;
      i++;
    }
    if(strcmp(ARGV[i], "-o") == 0 || strcmp(ARGV[i], "--output_file") == 0) {
      outfile_path = ARGV[i + 1];
      i++;
    }
    if(strcmp(ARGV[i], "--use_mpi_lbcast") == 0) {
        params.bcast_type = HPL_COMM_COLLECTIVE;
        use_mpi_lbcast = true;
    }
    if(strcmp(ARGV[i], "--use_mpi_allreduce_dmxswp") == 0) {
        params.allreduce_dmxswp_type = HPL_COMM_COLLECTIVE;
        use_mpi_allreduce_dmxswp = true;
    }
    if(strcmp(ARGV[i], "--use_mpi_allgatherv") == 0) {
        params.allgatherv_type = HPL_COMM_COLLECTIVE;
        use_mpi_allgatherv = true;
    }
    if(strcmp(ARGV[i], "--use_mpi_scatterv") == 0) {
        params.scatterv_type = HPL_COMM_COLLECTIVE;
        use_mpi_scatterv = true;
    }
  }

  /*
   * Check for enough processes in machine configuration
   */
  maxp = _P * _Q;
  if(maxp > size) {
    if(rank == 0)
      HPL_pwarn(stderr,
                __LINE__,
                "HPL_pdinfo",
                "Need at least %d processes for these tests",
                maxp);
    MPI_Finalize();
    exit(1);
  }

  /*
   * Split fraction
   */
  params.frac = frac;

  /*Node-local grid*/
  MPI_Comm nodeComm;
  MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);

  int localRank;
  int localSize;
  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_size(nodeComm, &localSize);

  if(_p < 1 && _q < 1) { // Neither p nor q specified
    _q = localSize;      // Assume a 1xq node-local grid
    _p = 1;
  } else if(_p < 1) { // q specified
    if(localSize % _q != 0) {
      if(rank == 0)
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "Node-local MPI grid cannot be split into q=%d columns",
                  _q);
      MPI_Finalize();
      exit(1);
    }
    _p = localSize / _q;
  } else if(_q < 1) { // p specified
    if(localSize % _p != 0) {
      if(rank == 0)
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "Node-local MPI grid cannot be split into p=%d rows",
                  _p);
      MPI_Finalize();
      exit(1);
    }
    _q = localSize / _p;
  } else {
    if(localSize != _p * _q) {
      if(rank == 0)
        HPL_pwarn(
            stderr, __LINE__, "HPL_pdinfo", "Invalid Node-local MPI grid");
      MPI_Finalize();
      exit(1);
    }
  }

  /*Check grid can be distributed to nodes*/
  if(_Q % _q != 0 || _P % _p != 0) {
    if(rank == 0)
      HPL_pwarn(stderr,
                __LINE__,
                "HPL_pdinfo",
                "MPI grid is not uniformly distributed amoung nodes, "
                "(P,Q)=(%d,%d) and (p,q)=(%d,%d)",
                _P,
                _Q,
                _p,
                _q);
    MPI_Finalize();
    exit(1);
  }
  MPI_Comm_free(&nodeComm);
  /*
   * Node-local Process grids, mapping
   */
  params.loc_proc_rows = _p;
  params.loc_proc_cols = _q;

  if(inputfile == false && cmdlinerun == true) {
    // We were given run paramters via the cmd line so skip
    // trying to read from an input file and just fill a
    // TEST structure.

    /*
     * Problem size (>=0) (N)
     */
    nS  = 1;
    N[0] = n;
    params.matrix_sizes.resize(1);
    params.matrix_sizes[0] = n;
    /*
     * Block size (>=1) (NB)
     */
    nBS  = 1;
    NB[0] = nb;
    params.bs.resize(1);
    params.bs[0] = nb;
    /*
     * Process grids, mapping, (>=1) (P, Q)
     */
    params.process_ordering = HPL_COLUMN_MAJOR;
    params.process_ordering = HPL_COLUMN_MAJOR;
    nPQS    = 1;
    P[0]     = _P;
    Q[0]     = _Q;
    params.gl_proc_rows.resize(1);
    params.gl_proc_cols.resize(1);
    params.gl_proc_rows[0] = _P;
    params.gl_proc_cols[0] = _Q;
    /*
     * Panel factorization algorithm (PF)
     */
    nPFS = 1;
    PF[0] = HPL_RIGHT_LOOKING; // HPL_LEFT_LOOKING, HPL_CROUT;
    params.panel_facts.resize(1);
    params.panel_facts[0] = HPL_RIGHT_LOOKING;
    /*
     * Recursive stopping criterium (>=1) (NBM)
     */
    nBMS  = 1;
    NBM[0] = 16;
    params.recursive_stop_crit.resize(1);
    params.recursive_stop_crit[0] = 16;
    /*
     * Number of panels in recursion (>=2) (NDV)
     */
    nDVS  = 1;
    NDV[0] = 2;
    params.num_panels_recursion.resize(1);
    params.num_panels_recursion[0] = 2;
    /*
     * Recursive panel factorization (RF)
     */
    nRFS = 1;
    RF[0] = HPL_RIGHT_LOOKING; // HPL_LEFT_LOOKING, HPL_CROUT;
    params.recursive_facts.resize(1);
    params.recursive_facts[0] = HPL_RIGHT_LOOKING;
    /*
     * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
     */
    nTPS = 1;
    TP[0] = HPL_1RING;
    params.bcast_algos.resize(1);
    params.bcast_algos[0] = HPL_1RING;
   
    if(!use_mpi_lbcast) 
        params.bcast_type = HPL_COMM_CUSTOM_IMPL;
    if(!use_mpi_allreduce_dmxswp)
        params.allreduce_dmxswp_type = HPL_COMM_CUSTOM_IMPL;
    if(!use_mpi_allgatherv)
        params.allgatherv_type = HPL_COMM_CUSTOM_IMPL;
    if(!use_mpi_scatterv)
        params.scatterv_type = HPL_COMM_CUSTOM_IMPL;
    /*
     * Lookahead depth (>=0) (NDH)
     */
    nDHS = 1;
    DH[0] = 1;
    params.lookahead_depths.resize(1);
    params.lookahead_depths[0] = 1;
    /*
     * Swapping algorithm (0,1 or 2) (FSWAP)
     */
    params.fswap = HPL_SWAP01;
    /*
     * Swapping threshold (>=0) (TSWAP)
     */
    params.swap_threshold_cols = 64;
    /*
     * L1 in (no-)transposed form (0 or 1)
     */
    L1NOTRAN = 1;
    params.L1_no_transpose = true;
    /*
     * U  in (no-)transposed form (0 or 1)
     */
    UNOTRAN = 0;
    params.U_no_transpose = false;
    /*
     * Equilibration (0=no, 1=yes)
     */
    EQUIL = 0;
    params.equil = false;
    /*
     * Memory alignment in bytes (> 0) (ALIGN)
     */
    params.mem_align = 8;

    /*
     * Compute and broadcast machine epsilon
     */
    TEST->epsil = HPL_pdlamch(MPI_COMM_WORLD, HPL_MACH_EPS);

    if(rank == 0) {
      if((TEST->outfp = fopen(outfile_path.c_str(), "w")) == NULL) { error = 1; }
    }
    (void)HPL_all_reduce((void*)(&error), 1, HPL_INT, HPL_MAX, MPI_COMM_WORLD);
    if(error) {
      if(rank == 0)
        HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "cannot open file HPL.out.");
      MPI_Finalize();
      exit(1);
    }
  } else {
    /*
     * Process 0 reads the input data, broadcasts to other processes and
     * writes needed information to TEST->outfp.
     */
    if(rank == 0) {
      /*
       * Open file and skip data file header
       */
      if((infp = fopen(inputFileName.c_str(), "r")) == NULL) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "cannot open file %s",
                  inputFileName.c_str());
        error = 1;
      }

      fgets(line, HPL_LINE_MAX - 2, infp);
      fgets(auth, HPL_LINE_MAX - 2, infp);
      /*
       * Read name and unit number for summary output file
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", file);
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      fid = atoi(num);
      if(fid == 6)
        TEST->outfp = stdout;
      else if(fid == 7)
        TEST->outfp = stderr;
      else if((TEST->outfp = fopen(file, "w")) == NULL) {
        HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "cannot open file %s.", file);
        error = 1;
      }
      /*
       * Read and check the parameter values for the tests.
       *
       * Problem size (>=0) (N)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int num_sizes{std::stoi(num)};
      nS = num_sizes;
      params.matrix_sizes.resize(num_sizes);
      if((nS < 1) || (nS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %d",
                  "Number of values of N is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }

      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.matrix_sizes[i] = std::stoi(num);
        if((N[i] = atoi(num)) < 0) {
          HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of N less than 0");
          error = 1;
        }
      }
      /*
       * Block size (>=1) (NB)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      //NBS = atoi(num);
      const int nbs{std::stoi(num)};
      params.bs.resize(nbs);
      nBS = nbs;
      if((nBS < 1) || (nBS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of NB is less than 1 or",
                  "greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }

      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nBS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.bs[i] = std::stoi(num);
        NB[i] = params.bs[i];
        if(NB[i] < 1) {
          HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of NB less than 1");
          error = 1;
        }
      }
      /*
       * Process grids, mapping, (>=1) (P, Q)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      params.process_ordering = (atoi(num) == 1 ? HPL_COLUMN_MAJOR : HPL_ROW_MAJOR);
      params.process_ordering = params.process_ordering;

      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int npqs{std::stoi(num)};
      params.gl_proc_rows.resize(npqs);
      params.gl_proc_cols.resize(npqs);
      nPQS = npqs;
      if((nPQS < 1) || (nPQS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of grids is less",
                  "than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }

      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nPQS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.gl_proc_rows[i] = std::stoi(num);
        P[i] = params.gl_proc_rows[i];
        if(params.gl_proc_rows[i] < 1) {
          HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of P less than 1");
          error = 1;
        }
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nPQS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.gl_proc_cols[i] = std::stoi(num);
        Q[i] = params.gl_proc_cols[i];
        if(params.gl_proc_cols[i] < 1) {
          HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of Q less than 1");
          error = 1;
        }
      }
      /*
       * Check for enough processes in machine configuration
       */
      maxp = 0;
      for(int i = 0; i < nPQS; i++) {
        nprocs = P[i] * Q[i];
        maxp   = Mmax(maxp, nprocs);
      }
      if(maxp > size) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "Need at least %d processes for these tests",
                  maxp);
        error = 1;
      }
      /*
       * Checking threshold value (TEST->thrsh)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      TEST->thrsh = atof(num);
      /*
       * Panel factorization algorithm (PF)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int npfs{std::stoi(num)};
      params.panel_facts.resize(npfs);
      nPFS = npfs;
      if((nPFS < 1) || (nPFS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "number of values of PFACT",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nPFS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        const int j = atoi(num);
        if(j == 0) {
          PF[i] = HPL_LEFT_LOOKING;
          params.panel_facts[i] = HPL_LEFT_LOOKING;
        }
        else if(j == 1) {
          PF[i] = HPL_CROUT;
          params.panel_facts[i] = HPL_CROUT;
        }
        else if(j == 2) {
          PF[i] = HPL_RIGHT_LOOKING;
          params.panel_facts[i] = HPL_RIGHT_LOOKING;
        }
        else {
          PF[i] = HPL_RIGHT_LOOKING;
          params.panel_facts[i] = HPL_RIGHT_LOOKING;
        }
      }
      /*
       * Recursive stopping criterium (>=1) (NBM)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int nbms{std::stoi(num)};
      params.recursive_stop_crit.resize(nbms);
      nBMS = nbms;
      if((nBMS < 1) || (nBMS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of NBMIN",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nBMS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.recursive_stop_crit[i] = std::stoi(num);
        NBM[i] = params.recursive_stop_crit[i];
        if(NBM[i] < 1) {
          HPL_pwarn(
              stderr, __LINE__, "HPL_pdinfo", "Value of NBMIN less than 1");
          error = 1;
        }
      }
      /*
       * Number of panels in recursion (>=2) (NDV)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int ndvs{std::stoi(num)};
      params.num_panels_recursion.resize(ndvs);
      nDVS = ndvs;
      if((nDVS < 1) || (nDVS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of NDIV",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nDVS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.num_panels_recursion[i] = std::stoi(num);
        NDV[i] = params.num_panels_recursion[i];
        if(NDV[i] < 2) {
          HPL_pwarn(
              stderr, __LINE__, "HPL_pdinfo", "Value of NDIV less than 2");
          error = 1;
        }
      }
      /*
       * Recursive panel factorization (RF)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int nrfs{std::stoi(num)};
      params.recursive_facts.resize(nrfs);
      nRFS = nrfs;
      if((nRFS < 1) || (nRFS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of RFACT",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nRFS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        const int j = atoi(num);
        if(j == 0) {
          RF[i] = HPL_LEFT_LOOKING;
          params.recursive_facts[i] = HPL_LEFT_LOOKING;
        }
        else if(j == 1) {
          RF[i] = HPL_CROUT;
          params.recursive_facts[i] = HPL_CROUT;
        }
        else if(j == 2) {
          RF[i] = HPL_RIGHT_LOOKING;
          params.recursive_facts[i] = HPL_RIGHT_LOOKING;
        }
        else {
          RF[i] = HPL_RIGHT_LOOKING;
          params.recursive_facts[i] = HPL_RIGHT_LOOKING;
        }
      }
      /*
       * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int ntps{std::stoi(num)};
      nTPS = ntps;
      if((nTPS < 1) || (nTPS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of BCAST",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nTPS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        const int j = atoi(num);
        if(j == 0) {
          TP[i] = HPL_1RING;
          params.bcast_algos[i] = HPL_1RING;
        }
        else if(j == 1) {
          TP[i] = HPL_1RING_M;
          params.bcast_algos[i] = HPL_1RING_M;
        }
        else if(j == 2) {
          TP[i] = HPL_2RING;
          params.bcast_algos[i] = HPL_2RING;
        }
        else if(j == 3) {
          TP[i] = HPL_2RING_M;
          params.bcast_algos[i] = HPL_2RING_M;
        }
        else if(j == 4) {
          TP[i] = HPL_BLONG;
          params.bcast_algos[i] = HPL_BLONG;
        }
        else { // if(j == 5)
          TP[i] = HPL_BLONG_M;
          params.bcast_algos[i] = HPL_BLONG_M;
        }
      }

      // Use collectives?
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if(!use_mpi_lbcast) 
        params.bcast_type = (atoi(num) == 1 ? HPL_COMM_COLLECTIVE : HPL_COMM_CUSTOM_IMPL);

      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if(!use_mpi_allreduce_dmxswp)
        params.allreduce_dmxswp_type = (atoi(num) == 1 ? HPL_COMM_COLLECTIVE : HPL_COMM_CUSTOM_IMPL);
      
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if(!use_mpi_allgatherv)
        params.allgatherv_type = (atoi(num) == 1 ? HPL_COMM_COLLECTIVE : HPL_COMM_CUSTOM_IMPL);
      
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if(!use_mpi_scatterv)
        params.scatterv_type = (atoi(num) == 1 ? HPL_COMM_COLLECTIVE : HPL_COMM_CUSTOM_IMPL);

      /*
       * Lookahead depth (>=0) (NDH)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int ndhs{std::stoi(num)};
      params.lookahead_depths.resize(ndhs);
      nDHS = ndhs;
      if((nDHS < 1) || (nDHS > HPL_MAX_PARAM)) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "%s %s %d",
                  "Number of values of DEPTH",
                  "is less than 1 or greater than",
                  HPL_MAX_PARAM);
        error = 1;
      }
      fgets(line, HPL_LINE_MAX - 2, infp);
      lineptr = line;
      for(int i = 0; i < nDHS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        params.lookahead_depths[i] = std::stoi(num);
        DH[i] = params.lookahead_depths[i];
        if(DH[i] < 0) {
          HPL_pwarn(
              stderr, __LINE__, "HPL_pdinfo", "Value of DEPTH less than 0");
          error = 1;
        }
        // NC: We require lookahead depth of 1
        if(DH[i] != 1) {
          HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of DEPTH must be 1");
          error = 1;
        }
      }
      /*
       * Swapping algorithm (0,1 or 2) (FSWAP)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int j = atoi(num);
      if(j == 0) {
        params.fswap = HPL_SWAP00;
      }
      else if(j == 1) {
        params.fswap = HPL_SWAP01;
      }
      else if(j == 2) {
        params.fswap = HPL_SW_MIX;
      }
      else {
        params.fswap = HPL_SWAP01;
      }
      // NC: Only one rowswapping algorithm implemented
      if(params.fswap != HPL_SWAP01) {
        HPL_pwarn(stderr, __LINE__, "HPL_pdinfo", "Value of SWAP must be 1");
        error = 1;
      }
      /*
       * Swapping threshold (>=0) (TSWAP)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int tswap{std::stoi(num)};
      params.swap_threshold_cols = tswap;
      if(params.swap_threshold_cols <= 0) {
          params.swap_threshold_cols = 0;
      }
      /*
       * L1 in (no-)transposed form (0 or 1)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int l1notran{std::stoi(num)};
      params.L1_no_transpose = (l1notran == 0) ? false : true;
      L1NOTRAN = l1notran;
      if((L1NOTRAN != 0) && (L1NOTRAN != 1)) {
          L1NOTRAN = 0;
      }
      /*
       * U  in (no-)transposed form (0 or 1)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int unotran{std::stoi(num)};
      params.U_no_transpose = (unotran == 0) ? false : true;
      UNOTRAN = unotran;
      if((UNOTRAN != 0) && (UNOTRAN != 1)) {
          UNOTRAN = 0;
      }

      // NC: We don't support holding U in no-transpose form anymore
      if(UNOTRAN != 0) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "U  in no-transposed form unsupported");
        error = 1;
      }
      /*
       * Equilibration (0=no, 1=yes)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      const int equil{std::stoi(num)};
      params.equil = (equil == 0) ? false : true;
      EQUIL = equil;
      if((EQUIL != 0) && (EQUIL != 1))
          EQUIL = 1;

      // NC: We don't currently support Equilibration
      if(EQUIL != 0) {
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "Equilibration currently unsupported");
        error = 1;
      }
      /*
       * Memory alignment in bytes (> 0) (ALIGN)
       */
      fgets(line, HPL_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      params.mem_align = std::stoi(num);
      if(params.mem_align <= 0) {
          params.mem_align = 4;
      }

      /*
       * Close input file
       */
      if(error) {
        (void)fclose(infp);
      }
    } else {
      TEST->outfp = NULL;
    }

    /*
     * Check for error on reading input file
     */
    (void)HPL_all_reduce((void*)(&error), 1, HPL_INT, HPL_MAX, MPI_COMM_WORLD);
    if(error) {
      if(rank == 0)
        HPL_pwarn(stderr,
                  __LINE__,
                  "HPL_pdinfo",
                  "Illegal input in file HPL.dat. Exiting ...");
      MPI_Finalize();
      exit(1);
    }
    /*
     * Compute and broadcast machine epsilon
     */
    TEST->epsil = HPL_pdlamch(MPI_COMM_WORLD, HPL_MACH_EPS);
    /*
     * Pack information arrays and broadcast
     */
    (void)HPL_broadcast(
        (void*)(&(TEST->thrsh)), 1, HPL_DOUBLE, 0, MPI_COMM_WORLD);
    /*
     * Broadcast array sizes
     */
    iwork = (int*)malloc((size_t)(19) * sizeof(int));
    if(rank == 0) {
      iwork[0]  = nS;
      iwork[1]  = nBS;
      iwork[2]  = (params.process_ordering == HPL_ROW_MAJOR ? 0 : 1);
      iwork[3]  = nPQS;
      iwork[4]  = nPFS;
      iwork[5]  = nBMS;
      iwork[6]  = nDVS;
      iwork[7]  = nRFS;
      iwork[8]  = nTPS;
      iwork[9]  = nDHS;
      iwork[10] = params.swap_threshold_cols;
      iwork[11] = L1NOTRAN;
      iwork[12] = UNOTRAN;
      iwork[13] = EQUIL;
      iwork[14] = params.mem_align;
      iwork[15] = static_cast<int>(params.bcast_type);
      iwork[16] = static_cast<int>(params.allreduce_dmxswp_type);
      iwork[17] = static_cast<int>(params.allgatherv_type);
      iwork[18] = static_cast<int>(params.scatterv_type);
    }
    (void)HPL_broadcast((void*)iwork, 19, HPL_INT, 0, MPI_COMM_WORLD);
    if(rank != 0) {
      nS       = iwork[0];
      nBS      = iwork[1];
      params.process_ordering  = (iwork[2] == 0 ? HPL_ROW_MAJOR : HPL_COLUMN_MAJOR);
      nPQS     = iwork[3];
      nPFS     = iwork[4];
      nBMS     = iwork[5];
      nDVS     = iwork[6];
      nRFS     = iwork[7];
      nTPS     = iwork[8];
      nDHS     = iwork[9];
      params.mem_align    = iwork[14];
      params.bcast_type = static_cast<HPL_Comm_impl_type>(iwork[15]);
      params.allreduce_dmxswp_type = static_cast<HPL_Comm_impl_type>(iwork[16]);
      params.allgatherv_type = static_cast<HPL_Comm_impl_type>(iwork[17]);
      params.scatterv_type = static_cast<HPL_Comm_impl_type>(iwork[18]);
      
      params.matrix_sizes.resize(iwork[0]);
      params.bs.resize(iwork[1]);
      params.gl_proc_rows.resize(iwork[3]);
      params.gl_proc_cols.resize(iwork[3]);
      params.panel_facts.resize(iwork[4]);
      params.recursive_stop_crit.resize(iwork[5]);
      params.num_panels_recursion.resize(iwork[6]);
      params.recursive_facts.resize(iwork[7]);
      params.bcast_algos.resize(iwork[8]);
      params.lookahead_depths.resize(iwork[9]);
      params.swap_threshold_cols = iwork[10];
      params.L1_no_transpose = iwork[11];
      params.U_no_transpose = iwork[12];
      params.equil = iwork[13];
    }
    if(iwork) free(iwork);
    /*
     * Pack information arrays and broadcast
     */
    lwork = (nS) + (nBS) + 2 * (nPQS) + (nPFS) + (nBMS) + (nDVS) +
            (nRFS) + (nTPS) + (nDHS) + 1;
    iwork = (int*)malloc((size_t)(lwork) * sizeof(int));
    if(rank == 0) {
      int j = 0;
      for(int i = 0; i < nS; i++) {
        iwork[j] = N[i];
        j++;
      }
      for(int i = 0; i < nBS; i++) {
        iwork[j] = NB[i];
        j++;
      }
      for(int i = 0; i < nPQS; i++) {
        iwork[j] = P[i];
        j++;
      }
      for(int i = 0; i < nPQS; i++) {
        iwork[j] = Q[i];
        j++;
      }
      for(int i = 0; i < nPFS; i++) {
        if(PF[i] == HPL_LEFT_LOOKING)
          iwork[j] = 0;
        else if(PF[i] == HPL_CROUT)
          iwork[j] = 1;
        else if(PF[i] == HPL_RIGHT_LOOKING)
          iwork[j] = 2;
        j++;
      }
      for(int i = 0; i < nBMS; i++) {
        iwork[j] = NBM[i];
        j++;
      }
      for(int i = 0; i < nDVS; i++) {
        iwork[j] = NDV[i];
        j++;
      }
      for(int i = 0; i < nRFS; i++) {
        if(RF[i] == HPL_LEFT_LOOKING)
          iwork[j] = 0;
        else if(RF[i] == HPL_CROUT)
          iwork[j] = 1;
        else if(RF[i] == HPL_RIGHT_LOOKING)
          iwork[j] = 2;
        j++;
      }
      for(int i = 0; i < nTPS; i++) {
        if(TP[i] == HPL_1RING)
          iwork[j] = 0;
        else if(TP[i] == HPL_1RING_M)
          iwork[j] = 1;
        else if(TP[i] == HPL_2RING)
          iwork[j] = 2;
        else if(TP[i] == HPL_2RING_M)
          iwork[j] = 3;
        else if(TP[i] == HPL_BLONG)
          iwork[j] = 4;
        else if(TP[i] == HPL_BLONG_M)
          iwork[j] = 5;
        j++;
      }
      for(int i = 0; i < nDHS; i++) {
        iwork[j] = DH[i];
        j++;
      }

      if(params.fswap == HPL_SWAP00)
        iwork[j] = 0;
      else if(params.fswap == HPL_SWAP01)
        iwork[j] = 1;
      else if(params.fswap == HPL_SW_MIX)
        iwork[j] = 2;
      j++;
    }
    (void)HPL_broadcast((void*)iwork, lwork, HPL_INT, 0, MPI_COMM_WORLD);
    if(rank != 0) {
      int j = 0;
      for(int i = 0; i < nS; i++) {
        N[i] = iwork[j];
        params.matrix_sizes[i] = iwork[j];
        j++;
      }
      for(int i = 0; i < nBS; i++) {
        NB[i] = iwork[j];
        params.bs[i] = iwork[j];
        j++;
      }
      for(int i = 0; i < nPQS; i++) {
        P[i] = iwork[j];
        params.gl_proc_rows[i] = iwork[j];
        j++;
      }
      for(int i = 0; i < nPQS; i++) {
        Q[i] = iwork[j];
        params.gl_proc_cols[i] = iwork[j];
        j++;
      }

      for(int i = 0; i < nPFS; i++) {
        if(iwork[j] == 0) {
          PF[i] = HPL_LEFT_LOOKING;
          params.panel_facts[i] = HPL_LEFT_LOOKING;
        }
        else if(iwork[j] == 1) {
          PF[i] = HPL_CROUT;
          params.panel_facts[i] = HPL_CROUT;
        }
        else if(iwork[j] == 2) {
          PF[i] = HPL_RIGHT_LOOKING;
          params.panel_facts[i] = HPL_RIGHT_LOOKING;
        }
        j++;
      }
      for(int i = 0; i < nBMS; i++) {
        NBM[i] = iwork[j];
        params.recursive_stop_crit[i] = iwork[j];
        j++;
      }
      for(int i = 0; i < nDVS; i++) {
        NDV[i] = iwork[j];
        params.num_panels_recursion[i] = iwork[j];
        j++;
      }
      for(int i = 0; i < nRFS; i++) {
        if(iwork[j] == 0) {
          RF[i] = HPL_LEFT_LOOKING;
          params.recursive_facts[i] = HPL_LEFT_LOOKING;
        }
        else if(iwork[j] == 1) {
          RF[i] = HPL_CROUT;
          params.recursive_facts[i] = HPL_CROUT;
        }
        else if(iwork[j] == 2) {
          RF[i] = HPL_RIGHT_LOOKING;
          params.recursive_facts[i] = HPL_RIGHT_LOOKING;
        }
        j++;
      }
      for(int i = 0; i < nTPS; i++) {
        if(iwork[j] == 0) {
          TP[i] = HPL_1RING;
          params.bcast_algos[i] = HPL_1RING;
        }
        else if(iwork[j] == 1) {
          TP[i] = HPL_1RING_M;
          params.bcast_algos[i] = HPL_1RING_M;
        }
        else if(iwork[j] == 2) {
          TP[i] = HPL_2RING;
          params.bcast_algos[i] = HPL_2RING;
        }
        else if(iwork[j] == 3) {
          TP[i] = HPL_2RING_M;
          params.bcast_algos[i] = HPL_2RING_M;
        }
        else if(iwork[j] == 4) {
          TP[i] = HPL_BLONG;
          params.bcast_algos[i] = HPL_BLONG;
        }
        else if(iwork[j] == 5) {
          TP[i] = HPL_BLONG_M;
          params.bcast_algos[i] = HPL_BLONG_M;
        }
        j++;
      }
      for(int i = 0; i < nDHS; i++) {
        DH[i] = iwork[j];
        params.lookahead_depths[i] = iwork[j];
        j++;
      }

      if(iwork[j] == 0) {
        params.fswap = HPL_SWAP00;
      }
      else if(iwork[j] == 1) {
        params.fswap = HPL_SWAP01;
      }
      else if(iwork[j] == 2) {
        params.fswap = HPL_SW_MIX;
      }
      j++;
    }
    if(iwork) free(iwork);
  }

  /*
   * regurgitate input
   */
  if(rank == 0) {
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "========================================",
                "========================================");
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "HPLinpack 2.2  --  High-Performance Linpack benchmark  --  ",
                " February 24, 2016");
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "Written by A. Petitet and R. Clint Whaley,  ",
                "Innovative Computing Laboratory, UTK");
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "Modified by Piotr Luszczek, ",
                "Innovative Computing Laboratory, UTK");
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "Modified by Julien Langou, ",
                "University of Colorado Denver");
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "========================================",
                "========================================");

    HPL_fprintf(TEST->outfp,
                "\n%s\n",
                "An explanation of the input/output parameters follows:");
    HPL_fprintf(TEST->outfp, "%s\n", "T/V    : Wall time / encoded variant.");
    HPL_fprintf(
        TEST->outfp, "%s\n", "N      : The order of the coefficient matrix A.");
    HPL_fprintf(
        TEST->outfp, "%s\n", "NB     : The partitioning blocking factor.");
    HPL_fprintf(TEST->outfp, "%s\n", "P      : The number of process rows.");
    HPL_fprintf(TEST->outfp, "%s\n", "Q      : The number of process columns.");
    HPL_fprintf(TEST->outfp,
                "%s\n",
                "Time   : Time in seconds to solve the linear system.");
    HPL_fprintf(TEST->outfp,
                "%s\n\n",
                "Gflops : Rate of execution for solving the linear system.");
    HPL_fprintf(
        TEST->outfp, "%s\n", "The following parameter values will be used:");
    /*
     * Problem size
     */
    HPL_fprintf(TEST->outfp, "\nN      :");
    for(int i = 0; i < Mmin(8, nS); i++) {
        HPL_fprintf(TEST->outfp, "%8d ", params.matrix_sizes[i]);
    }
    if(nS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nS); i++) {
          HPL_fprintf(TEST->outfp, "%8d ", params.matrix_sizes[i]);
      }
      if(nS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nS; i++)
            HPL_fprintf(TEST->outfp, "%8d ", params.matrix_sizes[i]);
      }
    }
    /*
     * Distribution blocking factor
     */
    HPL_fprintf(TEST->outfp, "\nNB     :");
    for(int i = 0; i < Mmin(8, nBS); i++) {
        HPL_fprintf(TEST->outfp, "%8d ", params.bs[i]);
    }
    if(nBS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nBS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.bs[i]);
      if(nBS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nBS; i++)
            HPL_fprintf(TEST->outfp, "%8d ", params.bs[i]);
      }
    }
    /*
     * Process mapping
     */
    HPL_fprintf(TEST->outfp, "\nPMAP   :");
    if(params.process_ordering == HPL_ROW_MAJOR)
      HPL_fprintf(TEST->outfp, " Row-major process mapping");
    else if(params.process_ordering == HPL_COLUMN_MAJOR)
      HPL_fprintf(TEST->outfp, " Column-major process mapping");
    /*
     * Process grid
     */
    HPL_fprintf(TEST->outfp, "\nP      :");
    for(int i = 0; i < Mmin(8, nPQS); i++) HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_rows[i]);
    if(nPQS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nPQS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_rows[i]);
      if(nPQS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nPQS; i++) HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_rows[i]);
      }
    }
    HPL_fprintf(TEST->outfp, "\nQ      :");
    for(int i = 0; i < Mmin(8, nPQS); i++) HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_cols[i]);
    if(nPQS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nPQS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_cols[i]);
      if(nPQS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nPQS; i++) HPL_fprintf(TEST->outfp, "%8d ", params.gl_proc_cols[i]);
      }
    }
    HPL_fprintf(TEST->outfp, "\np      : %8d.", params.loc_proc_rows);
    HPL_fprintf(TEST->outfp, "\nq      : %8d.", params.loc_proc_cols);
    /*
     * Panel Factorization
     */
    HPL_fprintf(TEST->outfp, "\nPFACT  :");
    for(int i = 0; i < Mmin(8, nPFS); i++) {
      if(params.panel_facts[i] == HPL_LEFT_LOOKING)
        HPL_fprintf(TEST->outfp, "    Left ");
      else if(params.panel_facts[i] == HPL_CROUT)
        HPL_fprintf(TEST->outfp, "   Crout ");
      else if(params.panel_facts[i] == HPL_RIGHT_LOOKING)
        HPL_fprintf(TEST->outfp, "   Right ");
    }
    if(nPFS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nPFS); i++) {
        if(params.panel_facts[i] == HPL_LEFT_LOOKING)
          HPL_fprintf(TEST->outfp, "    Left ");
        else if(params.panel_facts[i] == HPL_CROUT)
          HPL_fprintf(TEST->outfp, "   Crout ");
        else if(params.panel_facts[i] == HPL_RIGHT_LOOKING)
          HPL_fprintf(TEST->outfp, "   Right ");
      }
      if(nPFS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nPFS; i++) {
          if(params.panel_facts[i] == HPL_LEFT_LOOKING)
            HPL_fprintf(TEST->outfp, "    Left ");
          else if(params.panel_facts[i] == HPL_CROUT)
            HPL_fprintf(TEST->outfp, "   Crout ");
          else if(params.panel_facts[i] == HPL_RIGHT_LOOKING)
            HPL_fprintf(TEST->outfp, "   Right ");
        }
      }
    }
    /*
     * Recursive stopping criterium
     */
    HPL_fprintf(TEST->outfp, "\nNBMIN  :");
    for(int i = 0; i < Mmin(8, nBMS); i++)
      HPL_fprintf(TEST->outfp, "%8d ", params.recursive_stop_crit[i]);
    if(nBMS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nBMS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.recursive_stop_crit[i]);
      if(nBMS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nBMS; i++) HPL_fprintf(TEST->outfp, "%8d ", params.recursive_stop_crit[i]);
      }
    }
    /*
     * Number of panels in recursion
     */
    HPL_fprintf(TEST->outfp, "\nNDIV   :");
    for(int i = 0; i < Mmin(8, nDVS); i++)
      HPL_fprintf(TEST->outfp, "%8d ", params.num_panels_recursion[i]);
    if(nDVS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nDVS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.num_panels_recursion[i]);
      if(nDVS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nDVS; i++) HPL_fprintf(TEST->outfp, "%8d ", params.num_panels_recursion[i]);
      }
    }
    /*
     * Recursive Factorization
     */
    HPL_fprintf(TEST->outfp, "\nRFACT  :");
    for(int i = 0; i < Mmin(8, nRFS); i++) {
      if(params.recursive_facts[i] == HPL_LEFT_LOOKING)
        HPL_fprintf(TEST->outfp, "    Left ");
      else if(params.recursive_facts[i] == HPL_CROUT)
        HPL_fprintf(TEST->outfp, "   Crout ");
      else if(params.recursive_facts[i] == HPL_RIGHT_LOOKING)
        HPL_fprintf(TEST->outfp, "   Right ");
    }
    if(nRFS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nRFS); i++) {
        if(params.recursive_facts[i] == HPL_LEFT_LOOKING)
          HPL_fprintf(TEST->outfp, "    Left ");
        else if(params.recursive_facts[i] == HPL_CROUT)
          HPL_fprintf(TEST->outfp, "   Crout ");
        else if(params.recursive_facts[i] == HPL_RIGHT_LOOKING)
          HPL_fprintf(TEST->outfp, "   Right ");
      }
      if(nRFS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nRFS; i++) {
          if(params.recursive_facts[i] == HPL_LEFT_LOOKING)
            HPL_fprintf(TEST->outfp, "    Left ");
          else if(params.recursive_facts[i] == HPL_CROUT)
            HPL_fprintf(TEST->outfp, "   Crout ");
          else if(params.recursive_facts[i] == HPL_RIGHT_LOOKING)
            HPL_fprintf(TEST->outfp, "   Right ");
        }
      }
    }
    
    HPL_fprintf(TEST->outfp, "\nBCAST impl           :");
    if(params.bcast_type == HPL_COMM_CUSTOM_IMPL) {
        HPL_fprintf(TEST->outfp, " HPL custom ");
    } else {
        HPL_fprintf(TEST->outfp, " MPI collective ");
    }
    HPL_fprintf(TEST->outfp, "\nAllreduce_dmxswp impl:");
    if(params.allreduce_dmxswp_type == HPL_COMM_CUSTOM_IMPL) {
        HPL_fprintf(TEST->outfp, " HPL custom ");
    } else {
        HPL_fprintf(TEST->outfp, " MPI collective ");
    }
    HPL_fprintf(TEST->outfp, "\nScatter impl         :");
    if(params.scatterv_type == HPL_COMM_CUSTOM_IMPL) {
        HPL_fprintf(TEST->outfp, " HPL custom ");
    } else {
        HPL_fprintf(TEST->outfp, " MPI collective ");
    }
    HPL_fprintf(TEST->outfp, "\nAllgather impl       :");
    if(params.allgatherv_type == HPL_COMM_CUSTOM_IMPL) {
        HPL_fprintf(TEST->outfp, " HPL custom ");
    } else {
        HPL_fprintf(TEST->outfp, " MPI collective ");
    }

    /*
     * Broadcast topology
     */
    HPL_fprintf(TEST->outfp, "\nBCAST  :");
    for(int i = 0; i < Mmin(8, nTPS); i++) {
      if(params.bcast_algos[i] == HPL_1RING)
        HPL_fprintf(TEST->outfp, "   1ring ");
      else if(params.bcast_algos[i] == HPL_1RING_M)
        HPL_fprintf(TEST->outfp, "  1ringM ");
      else if(params.bcast_algos[i] == HPL_2RING)
        HPL_fprintf(TEST->outfp, "   2ring ");
      else if(params.bcast_algos[i] == HPL_2RING_M)
        HPL_fprintf(TEST->outfp, "  2ringM ");
      else if(params.bcast_algos[i] == HPL_BLONG)
        HPL_fprintf(TEST->outfp, "   Blong ");
      else if(params.bcast_algos[i] == HPL_BLONG_M)
        HPL_fprintf(TEST->outfp, "  BlongM ");
    }
    if(nTPS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nTPS); i++) {
        if(params.bcast_algos[i] == HPL_1RING)
          HPL_fprintf(TEST->outfp, "   1ring ");
        else if(params.bcast_algos[i] == HPL_1RING_M)
          HPL_fprintf(TEST->outfp, "  1ringM ");
        else if(params.bcast_algos[i] == HPL_2RING)
          HPL_fprintf(TEST->outfp, "   2ring ");
        else if(params.bcast_algos[i] == HPL_2RING_M)
          HPL_fprintf(TEST->outfp, "  2ringM ");
        else if(params.bcast_algos[i] == HPL_BLONG)
          HPL_fprintf(TEST->outfp, "   Blong ");
        else if(params.bcast_algos[i] == HPL_BLONG_M)
          HPL_fprintf(TEST->outfp, "  BlongM ");
      }
      if(nTPS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nTPS; i++) {
          if(params.bcast_algos[i] == HPL_1RING)
            HPL_fprintf(TEST->outfp, "   1ring ");
          else if(params.bcast_algos[i] == HPL_1RING_M)
            HPL_fprintf(TEST->outfp, "  1ringM ");
          else if(params.bcast_algos[i] == HPL_2RING)
            HPL_fprintf(TEST->outfp, "   2ring ");
          else if(params.bcast_algos[i] == HPL_2RING_M)
            HPL_fprintf(TEST->outfp, "  2ringM ");
          else if(params.bcast_algos[i] == HPL_BLONG)
            HPL_fprintf(TEST->outfp, "   Blong ");
          else if(params.bcast_algos[i] == HPL_BLONG_M)
            HPL_fprintf(TEST->outfp, "  BlongM ");
        }
      }
    }
    /*
     * Lookahead depths
     */
    HPL_fprintf(TEST->outfp, "\nDEPTH  :");
    for(int i = 0; i < Mmin(8, nDHS); i++) HPL_fprintf(TEST->outfp, "%8d ", params.lookahead_depths[i]);
    if(nDHS > 8) {
      HPL_fprintf(TEST->outfp, "\n        ");
      for(int i = 8; i < Mmin(16, nDHS); i++)
        HPL_fprintf(TEST->outfp, "%8d ", params.lookahead_depths[i]);
      if(nDHS > 16) {
        HPL_fprintf(TEST->outfp, "\n        ");
        for(int i = 16; i < nDHS; i++) HPL_fprintf(TEST->outfp, "%8d ", params.lookahead_depths[i]);
      }
    }
    /*
     * Swapping algorithm
     */
    HPL_fprintf(TEST->outfp, "\nSWAP   :");
    if(params.fswap == HPL_SWAP00)
      HPL_fprintf(TEST->outfp, " Binary-exchange");
    else if(params.fswap == HPL_SWAP01)
      HPL_fprintf(TEST->outfp, " Spread-roll (long)");
    else if(params.fswap == HPL_SW_MIX)
      HPL_fprintf(TEST->outfp, " Mix (threshold = %d)", params.swap_threshold_cols);
    /*
     * L1 storage form
     */
    HPL_fprintf(TEST->outfp, "\nL1     :");
    if(params.L1_no_transpose)
      HPL_fprintf(TEST->outfp, " no-transposed form");
    else
      HPL_fprintf(TEST->outfp, " transposed form");
    /*
     * U  storage form
     */
    HPL_fprintf(TEST->outfp, "\nU      :");
    if(params.U_no_transpose)
      HPL_fprintf(TEST->outfp, " no-transposed form");
    else
      HPL_fprintf(TEST->outfp, " transposed form");
    /*
     * Equilibration
     */
    HPL_fprintf(TEST->outfp, "\nEQUIL  :");
    if(params.equil)
      HPL_fprintf(TEST->outfp, " yes");
    else
      HPL_fprintf(TEST->outfp, " no");
    /*
     * Alignment
     */
    HPL_fprintf(TEST->outfp, "\nALIGN  : %d double precision words", params.mem_align);

    HPL_fprintf(TEST->outfp, "\n\n");
    /*
     * For testing only
     */
    if(TEST->thrsh > HPL_rzero) {
      HPL_fprintf(TEST->outfp,
                  "%s%s\n\n",
                  "----------------------------------------",
                  "----------------------------------------");
      HPL_fprintf(TEST->outfp,
                  "%s\n",
                  "- The matrix A is randomly generated for each test.");
      HPL_fprintf(TEST->outfp,
                  "%s\n",
                  "- The following scaled residual check will be computed:");
      HPL_fprintf(TEST->outfp,
                  "%s\n",
                  "      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || "
                  "b ||_oo ) * N )");
      HPL_fprintf(TEST->outfp,
                  "%s %21.6e\n",
                  "- The relative machine precision (eps) is taken to be     ",
                  TEST->epsil);
      HPL_fprintf(
          TEST->outfp,
          "%s   %11.1f\n\n",
          "- Computational tests pass if scaled residuals are less than      ",
          TEST->thrsh);
    }
  }
  return params;
}
