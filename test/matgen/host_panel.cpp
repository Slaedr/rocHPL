#include "alloc.hpp"

#include <hpl_auxil.hpp>
#include <hpl_pauxil.hpp>

namespace test {

void allocate_host_panel(const HPL_T_grid *const grid, const HPL_T_palg *const algo,
                         HPL_T_pmat *const mat,
                         const int nrows, const int gl_trailing_ncols, const int ncols,
                         const int gl_start_row, const int gl_start_col,
                         HPL_T_panel *const panel)
{
    panel->max_pinned_work_size = 0;
    panel->max_lwork_size       = 0;
    panel->max_uwork_size       = 0;
    panel->max_iwork_size       = 0;
    panel->max_fwork_size       = 0;
    panel->free_work_now        = 0;
    //panel->A                    = NULL;
    //panel->LWORK                = NULL;
    //panel->dLWORK               = NULL;
    //panel->UWORK                = NULL;
    //panel->dUWORK               = NULL;
    //panel->fWORK                = NULL;
    //panel->IWORK                = NULL;

    panel->grid = grid;
    panel->algo = algo;

    int icurcol, icurrow, ii, itmp1, jj, uwork, ml2, mycol, myrow, nb,
        npcol, nprow, nu, ldu;

    panel->grid = grid; /* ptr to the process grid */
    panel->algo = algo; /* ptr to the algo parameters */
    panel->pmat = mat;    /* ptr to the local array info */

    myrow = grid->myrow;
    mycol = grid->mycol;
    nprow = grid->nprow;
    npcol = grid->npcol;
    nb    = mat->nb;

    HPL_infog2l(gl_start_row, gl_start_col,
                nb, nb, nb, nb,
                0, 0,
                myrow, mycol,
                nprow, npcol,
                &ii, &jj,
                &icurrow, &icurcol);
    const int mp = HPL_numrocI(nrows, gl_start_row, nb, nb, myrow, 0, nprow);
    const int nq = HPL_numrocI(gl_trailing_ncols, gl_start_col, nb, nb, mycol, 0, npcol);

    const int inxtcol = MModAdd1(icurcol, npcol);
    const int inxtrow = MModAdd1(icurrow, nprow);

    /* ptr to trailing part of mat */
    panel->A  = mat->A;
    panel->dA = Mptr((double*)(mat->dA), ii, jj, mat->ld);

    /*
     * Workspace pointers are initialized to NULL.
     */
    panel->L2    = nullptr;
    panel->dL2   = nullptr;
    panel->L1    = nullptr;
    panel->dL1   = nullptr;
    panel->DINFO = nullptr;
    panel->U     = nullptr;
    panel->dU    = nullptr;
    panel->W     = nullptr;
    panel->dW    = nullptr;
    panel->U1    = nullptr;
    panel->dU1   = nullptr;
    panel->W1    = nullptr;
    panel->dW1   = nullptr;
    panel->U2    = nullptr;
    panel->dU2   = nullptr;
    panel->W2    = nullptr;
    panel->dW2   = nullptr;
    // panel->WORK    = NULL;
    // panel->IWORK   = NULL;
    /*
     * Local lengths, indexes process coordinates
     */
    panel->nb    = nb;      /* distribution blocking factor */
    panel->jb    = ncols;      /* panel width */
    panel->m     = nrows;       /* global # of rows of trailing part of mat */
    panel->n     = gl_trailing_ncols;       /* global # of cols of trailing part of A */
    panel->ia    = gl_start_row;      /* global row index of trailing part of A */
    panel->ja    = gl_start_col;      /* global col index of trailing part of A */
    panel->mp    = mp;      /* local # of rows of trailing part of A */
    panel->nq    = nq;      /* local # of cols of trailing part of A */
    panel->ii    = ii;      /* local row index of trailing part of A */
    panel->jj    = jj;      /* local col index of trailing part of A */
    panel->lda   = mat->ld;   /* local leading dim of array A */
    panel->dlda  = mat->ld;   /* local leading dim of array A */
    panel->prow  = icurrow; /* proc row owning 1st row of trailing A */
    panel->pcol  = icurcol; /* proc col owning 1st col of trailing A */
    panel->msgid = 0;     /* message id to be used for panel bcast */
                            /*
                             * Initialize  ldl2 and len to temporary dummy values and Update tag for
                             * next panel
                             */
    panel->ldl2  = 0;       /* local leading dim of array L2 */
    panel->dldl2 = 0;       /* local leading dim of array L2 */
    panel->len   = 0;       /* length of the buffer to broadcast */
    panel->nu0   = 0;
    panel->nu1   = 0;
    panel->nu2   = 0;
    panel->ldu0  = 0;
    panel->ldu1  = 0;
    panel->ldu2  = 0;

    /*
     * Figure out the exact amount of workspace  needed by the factorization
     * and the update - Allocate that space - Finish the panel data structu-
     * re initialization.
     *
     * L1:    ncols x ncols in all processes
     * DINFO: 1       in all processes
     *
     * We also make an array of necessary intergers for swaps in the update.
     *
     * If nprow is 1, we just allocate an array of 2*ncols integers for the swap.
     * When nprow > 1, we allocate the space for the index arrays immediate-
     * ly. The exact size of this array depends on the swapping routine that
     * will be used, so we allocate the maximum:
     *
     *       lindxU   is of size         ncols +
     *       lindxA   is of size at most ncols +
     *       lindxAU  is of size at most ncols +
     *       permU    is of size at most ncols
     *
     *       ipiv     is of size at most ncols
     *
     * that is  5*ncols.
     *
     * We make sure that those three arrays are contiguous in memory for the
     * later panel broadcast (using type punning to put the integer array at
     * the end.  We  also  choose  to put this amount of space right after
     * L2 (when it exist) so that one can receive a contiguous buffer.
     */

    /*Split fraction*/
    const double fraction = algo->frac;

    const size_t dalign      = algo->align * sizeof(double);
    const size_t lpiv = (5 * ncols * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

    if(npcol > 1) {
      ml2 = (myrow == icurrow ? mp - ncols : mp);
      ml2 = Mmax(0, ml2);
      ml2 = ((ml2 + 95) / 128) * 128 + 32; /*pad*/
    } else {
      ml2 = 0; // L2 is aliased inside A
    }

    /* Size of LBcast message */
    panel->len = ml2 * ncols + ncols * ncols + lpiv; // L2, L1, integer arrays

    /* space for L */
    int lwork = panel->len + 1;

    nu  = Mmax(0, (mycol == icurcol ? nq - ncols : nq));
    ldu = nu + ncols + 256; /*extra space for potential padding*/

    /* space for U */
    uwork = ncols * ldu;

    if(panel->max_lwork_size < (size_t)(lwork) * sizeof(double)) {
      //if(panel->LWORK) {
      //  free(panel->LWORK);
      //}
      // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) +
      // (size_t)4095)/(size_t)4096)*(size_t)4096;
      const size_t numbytes = (size_t)(lwork) * sizeof(double);

      //if(deviceMalloc(grid, (void**)&(panel->dLWORK), numbytes) != HPL_SUCCESS) {
      //  HPL_pabort(__LINE__,
      //             "HPL_pdpanel_init",
      //             "Device memory allocation failed for L workspace.");
      //}
      if(HPL_malloc((void**)&(panel->LWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for L workspace.");
      }

      panel->max_lwork_size = (size_t)(lwork) * sizeof(double);
    }
    if(panel->max_uwork_size < (size_t)(uwork) * sizeof(double)) {
      //if(panel->UWORK) {
      //  free(panel->UWORK);
      //}
      // size_t numbytes = (((size_t)((size_t)(uwork) * sizeof( double )) +
      // (size_t)4095)/(size_t)4096)*(size_t)4096;
      size_t numbytes = (size_t)(uwork) * sizeof(double);

      //if(deviceMalloc(grid, (void**)&(panel->dUWORK), numbytes) != HPL_SUCCESS) {
      //  HPL_pabort(__LINE__,
      //             "HPL_pdpanel_init",
      //             "Device memory allocation failed for U workspace.");
      //}
      if(HPL_malloc((void**)&(panel->UWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for U workspace.");
      }

      panel->max_uwork_size = (size_t)(uwork) * sizeof(double);
    }

    /*
     * Initialize the pointers of the panel structure
     */
    if(npcol == 1) {
      panel->L2    = panel->A + (myrow == icurrow ? ncols : 0);
      panel->dL2   = panel->dA + (myrow == icurrow ? ncols : 0);
      panel->ldl2  = mat->ld;
      panel->dldl2 = mat->ld; /*L2 is aliased inside A*/

      panel->L1  = (double*)panel->LWORK;
      panel->dL1 = (double*)panel->dLWORK;
    } else {
      panel->L2    = (double*)panel->LWORK;
      panel->dL2   = (double*)panel->dLWORK;
      panel->ldl2  = Mmax(0, ml2);
      panel->dldl2 = Mmax(0, ml2);

      panel->L1  = panel->L2 + ml2 * ncols;
      panel->dL1 = panel->dL2 + ml2 * ncols;
    }

    panel->U  = (double*)panel->UWORK;
    panel->dU = (double*)panel->dUWORK;
    panel->W  = mat->W;
    panel->dW = mat->dW;

    if(nprow == 1) {
      panel->nu0  = (mycol == inxtcol) ? Mmin(ncols, nu) : 0;
      panel->ldu0 = panel->nu0;

      panel->nu1  = 0;
      panel->ldu1 = 0;

      panel->nu2  = nu - panel->nu0;
      panel->ldu2 = ((panel->nu2 + 95) / 128) * 128 + 32; /*pad*/

      panel->U2  = panel->U + ncols * ncols;
      panel->dU2 = panel->dU + ncols * ncols;
      panel->U1  = panel->U2 + panel->ldu2 * ncols;
      panel->dU1 = panel->dU2 + panel->ldu2 * ncols;

      panel->permU  = (int*)(panel->L1 + ncols * ncols);
      panel->dpermU = (int*)(panel->dL1 + ncols * ncols);
      panel->ipiv   = panel->permU + ncols;
      panel->dipiv  = panel->dpermU + ncols;

      panel->DINFO  = (double*)(panel->ipiv + 2 * ncols);
      panel->dDINFO = (double*)(panel->dipiv + 2 * ncols);
    } else {
      const int NSplit = Mmax(0, ((((int)(mat->nq * fraction)) / nb) * nb));
      panel->nu0       = (mycol == inxtcol) ? Mmin(ncols, nu) : 0;
      panel->ldu0      = panel->nu0;

      panel->nu2  = Mmin(nu - panel->nu0, NSplit);
      panel->ldu2 = ((panel->nu2 + 95) / 128) * 128 + 32; /*pad*/

      panel->nu1  = nu - panel->nu0 - panel->nu2;
      panel->ldu1 = ((panel->nu1 + 95) / 128) * 128 + 32; /*pad*/

      panel->U2  = panel->U + ncols * ncols;
      panel->dU2 = panel->dU + ncols * ncols;
      panel->U1  = panel->U2 + panel->ldu2 * ncols;
      panel->dU1 = panel->dU2 + panel->ldu2 * ncols;

      panel->W2  = panel->W + ncols * ncols;
      panel->dW2 = panel->dW + ncols * ncols;
      panel->W1  = panel->W2 + panel->ldu2 * ncols;
      panel->dW1 = panel->dW2 + panel->ldu2 * ncols;

      panel->lindxA   = (int*)(panel->L1 + ncols * ncols);
      panel->dlindxA  = (int*)(panel->dL1 + ncols * ncols);
      panel->lindxAU  = panel->lindxA + ncols;
      panel->dlindxAU = panel->dlindxA + ncols;
      panel->lindxU   = panel->lindxAU + ncols;
      panel->dlindxU  = panel->dlindxAU + ncols;
      panel->permU    = panel->lindxU + ncols;
      panel->dpermU   = panel->dlindxU + ncols;

      // Put ipiv array at the end
      panel->ipiv  = panel->permU + ncols;
      panel->dipiv = panel->dpermU + ncols;

      panel->DINFO  = ((double*)panel->lindxA) + lpiv;
      panel->dDINFO = ((double*)panel->dlindxA) + lpiv;
    }

    *(panel->DINFO) = 0.0;

    /*
     * If nprow is 1, we just allocate an array of ncols integers to store the
     * pivot IDs during factoring, and a scratch array of mp integers.
     * When nprow > 1, we allocate the space for the index arrays immediate-
     * ly. The exact size of this array depends on the swapping routine that
     * will be used, so we allocate the maximum:
     *
     *    IWORK[0] is of size at most 1      +
     *    IPL      is of size at most 1      +
     *    IPID     is of size at most 4 * ncols +
     *    IPIV     is of size at most ncols     +
     *    SCRATCH  is of size at most MP
     *
     *    ipA      is of size at most 1      +
     *    iplen    is of size at most NPROW  + 1 +
     *    ipcounts is of size at most NPROW  +
     *    ioffsets is of size at most NPROW  +
     *    iwork    is of size at most MAX( 2*ncols, NPROW+1 ).
     *
     * that is  mp + 4 + 5*ncols + 3*NPROW + MAX( 2*ncols, NPROW+1 ).
     *
     * We use the fist entry of this to work array  to indicate  whether the
     * the  local  index arrays have already been computed,  and if yes,  by
     * which function:
     *    IWORK[0] = -1: no index arrays have been computed so far;
     *    IWORK[0] =  1: HPL_pdlaswp already computed those arrays;
     * This allows to save some redundant and useless computations.
     */
    if(nprow == 1) {
      lwork = mp + ncols;
    } else {
      itmp1 = (ncols << 1);
      lwork = nprow + 1;
      itmp1 = Mmax(itmp1, lwork);
      lwork = mp + 4 + (5 * ncols) + (3 * nprow) + itmp1;
    }

    if(panel->max_iwork_size < (size_t)(lwork) * sizeof(int)) {
      const size_t numbytes = (size_t)(lwork) * sizeof(int);

      if(HPL_malloc((void**)&(panel->IWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for integer workspace.");
      }
      panel->max_iwork_size = (size_t)(lwork) * sizeof(int);
    }

    if(lwork) {
      *(panel->IWORK) = -1;
    }

    /*Finally, we need 4 + 4*ncols entries of scratch for pdfact */
    lwork = (size_t)(((4 + ((unsigned int)(ncols) << 1)) << 1));
    if(panel->max_fwork_size < (size_t)(lwork) * sizeof(double)) {
      const size_t numbytes = (size_t)(lwork) * sizeof(double);

      if(HPL_malloc((void**)&(panel->fWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for pdfact scratch workspace.");
      }
      printf("Allocated %lu bytes in fwork.\n", numbytes); fflush(stdout);
      panel->max_fwork_size = (size_t)(lwork) * sizeof(double);
    }
}

void free_host_panel(HPL_T_panel *const PANEL) {
    if(PANEL->pmat->info == 0) {
        PANEL->pmat->info = *(PANEL->DINFO);
    }

    if(PANEL->free_work_now == 1) {
        if(PANEL->LWORK) {
          free(PANEL->LWORK);
        }
        if(PANEL->UWORK) {
          free(PANEL->UWORK);
        }

        PANEL->max_lwork_size = 0;
        PANEL->max_uwork_size = 0;

        if(PANEL->IWORK) {
          free(PANEL->IWORK);
        }
        if(PANEL->fWORK) {
          free(PANEL->fWORK);
        }

        PANEL->max_iwork_size = 0;
        PANEL->max_fwork_size = 0;
    }
}

}
