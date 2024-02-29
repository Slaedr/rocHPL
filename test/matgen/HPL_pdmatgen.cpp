/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.3 - December 2, 2018                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */

#include "matrix.hpp"

#include <iostream>
#include <omp.h>

#include <hpl_auxil.hpp>
#include <hpl_pauxil.hpp>

#include "matgen.hpp"
#include "../utils/error_handling.hpp"

namespace test {

int HPL_host_pdmat_init(const HPL_T_grid* GRID, const int N, const int ncols, const int NB,
                        HPL_T_pmat *const mat)
{
    int mycol, myrow, npcol, nprow, info[3];
    (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

    mat->n    = N;
    mat->nb   = NB;
    mat->info = 0;
    mat->mp   = HPL_numroc(N, NB, NB, myrow, 0, nprow);
    const int nq = HPL_numroc(ncols, NB, NB, mycol, 0, npcol);
    /*
     * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
     * N by N+1.  One column is added in every process column for the solve.
     * The  result  however  is stored in a 1 x N vector replicated in every
     * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
     * workspace is mp.
     */
    mat->ld = Mmax(1, mat->mp);
    mat->ld = ((mat->ld + 95) / 128) * 128 + 32; /*pad*/

    mat->nq = nq + 1;

    mat->dA = nullptr;
    mat->dX = nullptr;

    mat->dW = nullptr;
    mat->W  = nullptr;

    /*
     * Allocate dynamic memory
     */
    const size_t numbytes = (size_t)(mat->ld) * (size_t)(mat->nq) * sizeof(double);

    if((myrow == 0) && (mycol == 0)) {
        printf("Global matrix rows = %d, matrix cols = %d.\n", N, ncols);
        printf("Local matrix size = %g GBs\n",
               ((double)numbytes) / (1024 * 1024 * 1024));
    }

    if(HPL_malloc((void**)&(mat->A), numbytes) != HPL_SUCCESS) {
        HPL_pwarn(stdout,
                  __LINE__, "HPL_pdmatgen", "[%d,%d] %s",
                  info[1],
                  info[2],
                  "Panel memory allocation failed. Skip.");
        return HPL_FAILURE;
    }

#pragma omp parallel
    {
        /*First touch*/
        const int thread_rank = omp_get_thread_num();
        const int thread_size = omp_get_num_threads();

        for(int i = 0; i < mat->ld; i += NB) {
            if((i / NB) % thread_size == thread_rank) {
                const int mm = std::min(NB, mat->ld - i);
                for(int k = 0; k < NB; ++k) {
                    for(int j = 0; j < mm; ++j) {
                        mat->A[j + i + static_cast<size_t>(mat->ld) * k] = 0.0;
                    }
                }
            }
        }
    }

    int Anp;
    Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);
    if(Anp != mat->mp) {
        std::printf("ERROR!! Anp != mp!\n"); fflush(stdout);
    }

    size_t dworkspace_size = 0;
    size_t workspace_size  = 0;

    /*pdtrsv needs two vectors for B and W (and X on host) */
    dworkspace_size = Mmax(2 * Anp * sizeof(double), dworkspace_size);
    workspace_size  = Mmax((2 * Anp + nq) * sizeof(double), workspace_size);

    /*Scratch space for rows in pdlaswp (with extra space for padding) */
    dworkspace_size =
        Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), dworkspace_size);
    workspace_size =
        Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), workspace_size);

    if((myrow == 0) && (mycol == 0)) {
        printf("Local matrix workspace size = %g GBs\n",
               ((double)workspace_size) / (1024 * 1024 * 1024));
    }

    if(HPL_malloc((void**)&(mat->W), workspace_size) != HPL_SUCCESS) {
        HPL_pwarn(stdout,
                  __LINE__, "HPL_pdmatgen", "[%d,%d] %s",
                  info[1],
                  info[2],
                  "Host memory allocation failed for U workspace. Skip.");
        return HPL_FAILURE;
    }

    return HPL_SUCCESS;
}

void HPL_host_matfree(HPL_T_pmat *const mat)
{
    if(mat->A) {
        free(mat->A);
        mat->A = nullptr;
    }
    if(mat->W) {
        free(mat->W);
        mat->W = nullptr;
    }
}

void generate_random_values_host(HPL_T_pmat *const mat, const int seed)
{
    matgen::HPL_dmatgen(mat->mp, mat->nq, mat->A, mat->ld, seed);
}

mat_diff compare_matrices_host(const HPL_T_pmat *m1, const HPL_T_pmat *m2, const double reltol)
{
    mat_diff diff{true, true, -1, -1, 1.0};

    if(m1->mp != m2->mp) {
        diff.match_mp = false;
    }
    if(m1->nb != m2->nb) {
        diff.match_nb = false;
    }
    if(!diff.match_mp || !diff.match_nb) {
        return diff;
    }

    double diff_norm = 0;
    double base_norm = 0;

    for(int j = 0; j < m1->nb; j++) {
        for(int i = 0; i < m1->mp; i++) {
            const double delta = std::abs(m1->A[i + j*m1->ld] - m2->A[i + j*m2->ld]);
            const double base = std::max(std::abs(m1->A[i + j*m1->ld]), std::abs(m2->A[i + j*m2->ld]));
            if(delta/base > reltol) {
                diff.iA = i;
                diff.jA = j;
            }
            diff_norm = std::max(diff_norm, delta);
            base_norm = std::max(base_norm, base);
        }
    }

    diff.rel_diff_norm = diff_norm/base_norm;

    return diff;
}

void test_mat_same_host(const HPL_T_pmat *const m1, const HPL_T_pmat *const m2, const double reltol)
{
    if(m1 == m2) {
        std::cout << "WARNING: The two matrices point to the same one!" << std::endl;
        return;
    }
    const mat_diff diff = compare_matrices_host(m1, m2, reltol);
    std::cout << "Rel. difference norm between matrices = " << diff.rel_diff_norm << std::endl;
    if(!diff.match_mp || !diff.match_nb) {
        throw PreconditionNotMet("Matrix dimensions are different!");
    }
    if(diff.iA >= 0) {
        throw PreconditionNotMet("Matrix values are different at " + std::to_string(diff.iA)
                                 + ", " + std::to_string(diff.jA));
    }
}


}
