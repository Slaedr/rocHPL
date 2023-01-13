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
#include "hpl_exceptions.hpp"

int HPL_grid_info(const HPL_T_grid* GRID,
                  int*              NPROW,
                  int*              NPCOL,
                  int*              MYROW,
                  int*              MYCOL) {
  /*
   * Purpose
   * =======
   *
   * HPL_grid_info returns  the grid shape and the coordinates in the grid
   * of the calling process.  Successful  completion  is  indicated by the
   * returned error code  MPI_SUCCESS. Other error codes depend on the MPI
   * implementation.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * NPROW   (global output)               int *
   *         On exit,   NPROW  specifies the number of process rows in the
   *         grid. NPROW is at least one.
   *
   * NPCOL   (global output)               int *
   *         On exit,   NPCOL  specifies  the number of process columns in
   *         the grid. NPCOL is at least one.
   *
   * MYROW   (global output)               int *
   *         On exit,  MYROW  specifies my  row process  coordinate in the
   *         grid. MYROW is greater than or equal  to zero  and  less than
   *         NPROW.
   *
   * MYCOL   (global output)               int *
   *         On exit,  MYCOL specifies my column process coordinate in the
   *         grid. MYCOL is greater than or equal  to zero  and  less than
   *         NPCOL.
   *
   * ---------------------------------------------------------------------
   */

  *NPROW = GRID->nprow;
  *NPCOL = GRID->npcol;
  *MYROW = GRID->myrow;
  *MYCOL = GRID->mycol;
  return (MPI_SUCCESS);
}

/**
 * Returns the MPI rank of a process given its coordinates in the 2D process grid.
 */
int get_mpi_rank(const HPL_T_grid *const grid, const int procrow, const int proccol)
{
    const int p = grid->local_nprow;
    const int q = grid->local_npcol;
    const int local_size = p * q;

    const int noderow = procrow / p;
    const int local_myrow = procrow % p;
    const int nodecol = proccol / q;
    const int local_mycol = proccol % q;

    if(grid->order == HPL_COLUMN_MAJOR) {
        const int inode = nodecol * (grid->nprow / p) + noderow;
        const int local_rank = local_mycol * p + local_myrow;
        return inode * local_size + local_rank;
    } else {
        ORNL_HPL_THROW_NOT_IMPLEMENTED("MPI rank from 2D proc coords for row-major grid");
        return -1;
    }
}

