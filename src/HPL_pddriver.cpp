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

int main(int ARGC, char** ARGV) {
  /*
   * Purpose
   * =======
   *
   * main is the main driver program for testing the HPL routines.
   * This  program is  driven  by  a short data file named  "HPL.dat".
   *
   * ---------------------------------------------------------------------
   */
  HPL_T_grid grid;
  HPL_T_palg algo;
  HPL_T_test test;
  int mycol, myrow, npcol, nprow, rank, size;

  MPI_Init(&ARGC, &ARGV);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Op_create(HPL_dmxswp, true, &HPL_DMXSWP);

  /*
   * Read and check validity of test parameters from input file
   *
   * HPL Version 1.0, Linpack benchmark input file
   * Your message here
   * HPL.out      output file name (if any)
   * 6            device out (6=stdout,7=stderr,file)
   * 4            # of problems sizes (N)
   * 29 30 34 35  Ns
   * 4            # of NBs
   * 1 2 3 4      NBs
   * 0            PMAP process mapping (0=Row-,1=Column-major)
   * 3            # of process grids (P x Q)
   * 2 1 4        Ps
   * 2 4 1        Qs
   * 16.0         threshold
   * 3            # of panel fact
   * 0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
   * 2            # of recursive stopping criterium
   * 2 4          NBMINs (>= 1)
   * 1            # of panels in recursion
   * 2            NDIVs
   * 3            # of recursive panel fact.
   * 0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
   * 1            # of broadcast
   * 0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
   * 1            # of lookahead depth
   * 0            DEPTHs (>=0)
   * 2            SWAP (0=bin-exch,1=long,2=mix)
   * 4            swapping threshold
   * 0            L1 in (0=transposed,1=no-transposed) form
   * 0            U  in (0=transposed,1=no-transposed) form
   * 1            Equilibration (0=no,1=yes)
   * 8            memory alignment in double (> 0)
   */
  const HPL_Test_params params = HPL_pdinfo(ARGC, ARGV, &test);

  /*
   * Loop over different process grids - Define process grid. Go to bottom
   * of process grid loop if this case does not use my process.
   */
  for(size_t ipq = 0; ipq < params.gl_proc_rows.size(); ipq++) {
    (void)HPL_grid_init(
        MPI_COMM_WORLD, params.process_ordering, params.gl_proc_rows[ipq], params.gl_proc_cols[ipq],
        params.loc_proc_rows, params.loc_proc_cols, &grid);
    (void)HPL_grid_info(&grid, &nprow, &npcol, &myrow, &mycol);

    if((myrow < 0) || (myrow >= nprow) || (mycol < 0) || (mycol >= npcol)) {
        continue;
    }

    // Initialize GPU
    HPL_InitGPU(&grid);

    for(auto mat_size : params.matrix_sizes) {
      for(auto block_size : params.bs) { /* Loop over various blocking factors */
        for(auto ldh : params.lookahead_depths) {
          for(auto bcast_algo : params.bcast_algos) {
            for(auto rfact : params.recursive_facts) { /* Loop over various recursive factorizations */
              for(auto pfact : params.panel_facts) { /* Loop over various panel factorizations */
                for(auto rstop : params.recursive_stop_crit) {
                  for(auto npanelrecurse : params.num_panels_recursion) {
                    //algo.btopo = topval[itop];
                    //algo.depth = ndhval[indh];
                    //algo.nbmin = nbmval[inbm];
                    //algo.nbdiv = ndvval[indv];
                    algo.btopo = bcast_algo;
                    algo.depth = ldh;
                    algo.nbmin = rstop;
                    algo.nbdiv = npanelrecurse;
                    algo.comm_impls_types.bcast_type = params.bcast_type;
                    algo.comm_impls_types.allreduce_dmxswp_type = params.allreduce_dmxswp_type;
                    algo.comm_impls_types.allgatherv_type = params.allgatherv_type;
                    algo.comm_impls_types.scatterv_type = params.scatterv_type;

                    algo.pfact = pfact;

                    if(params.L1_no_transpose) {
                      if(pfact == HPL_LEFT_LOOKING)
                        algo.pffun = HPL_pdpanllN;
                      else if(pfact == HPL_CROUT)
                        algo.pffun = HPL_pdpancrN;
                      else
                        algo.pffun = HPL_pdpanrlN;

                      algo.rfact = rfact;
                      if(rfact == HPL_LEFT_LOOKING)
                        algo.rffun = HPL_pdrpanllN;
                      else if(rfact == HPL_CROUT)
                        algo.rffun = HPL_pdrpancrN;
                      else
                        algo.rffun = HPL_pdrpanrlN;

                      algo.upfun = HPL_pdupdateNT;
                    } else {
                      if(pfact == HPL_LEFT_LOOKING)
                        algo.pffun = HPL_pdpanllT;
                      else if(pfact == HPL_CROUT)
                        algo.pffun = HPL_pdpancrT;
                      else
                        algo.pffun = HPL_pdpanrlT;

                      algo.rfact = rfact;
                      if(rfact == HPL_LEFT_LOOKING)
                        algo.rffun = HPL_pdrpanllT;
                      else if(rfact == HPL_CROUT)
                        algo.rffun = HPL_pdrpancrT;
                      else
                        algo.rffun = HPL_pdrpanrlT;

                      algo.upfun = HPL_pdupdateTT;
                    }

                    algo.fswap = params.fswap;
                    algo.fsthr = params.swap_threshold_cols;
                    algo.equil = params.equil;
                    algo.align = params.mem_align;

                    algo.frac = params.frac;

                    HPL_pdtest(&test, &grid, &algo, mat_size, block_size);
                  }
                }
              }
            }
          }
        }
      }
    }
    (void)HPL_grid_exit(&grid);
    HPL_FreeGPU();
  }
  /*
   * Print ending messages, close output file, exit.
   */
  if(rank == 0) {
    test.ktest = test.kpass + test.kfail + test.kskip;
#ifndef HPL_DETAILED_TIMING
    HPL_fprintf(test.outfp,
                "%s%s\n",
                "========================================",
                "========================================");
#else
    if(test.thrsh > HPL_rzero)
      HPL_fprintf(test.outfp,
                  "%s%s\n",
                  "========================================",
                  "========================================");
#endif

    HPL_fprintf(test.outfp,
                "\n%s %6d %s\n",
                "Finished",
                test.ktest,
                "tests with the following results:");
    if(test.thrsh > HPL_rzero) {
      HPL_fprintf(test.outfp,
                  "         %6d %s\n",
                  test.kpass,
                  "tests completed and passed residual checks,");
      HPL_fprintf(test.outfp,
                  "         %6d %s\n",
                  test.kfail,
                  "tests completed and failed residual checks,");
      HPL_fprintf(test.outfp,
                  "         %6d %s\n",
                  test.kskip,
                  "tests skipped because of illegal input values.");
    } else {
      HPL_fprintf(test.outfp,
                  "         %6d %s\n",
                  test.kpass,
                  "tests completed without checking,");
      HPL_fprintf(test.outfp,
                  "         %6d %s\n",
                  test.kskip,
                  "tests skipped because of illegal input values.");
    }

    HPL_fprintf(test.outfp,
                "%s%s\n",
                "----------------------------------------",
                "----------------------------------------");
    HPL_fprintf(test.outfp, "\nEnd of Tests.\n");
    HPL_fprintf(test.outfp,
                "%s%s\n",
                "========================================",
                "========================================");

    if((test.outfp != stdout) && (test.outfp != stderr))
      (void)fclose(test.outfp);
  }

  MPI_Finalize();

  return (0);
}
