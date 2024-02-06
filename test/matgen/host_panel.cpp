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

    int icurcol, icurrow, ii, jj;

    panel->grid = grid; /* ptr to the process grid */
    panel->algo = algo; /* ptr to the algo parameters */
    panel->pmat = mat;    /* ptr to the local array info */

    const int myrow = grid->myrow;
    const int mycol = grid->mycol;
    const int nprow = grid->nprow;
    const int npcol = grid->npcol;
    const int nb    = mat->nb;

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

    const HPL_panel_sizes psz = get_panel_sizes(mat, nrows, gl_trailing_ncols, ncols,
                                                gl_start_row, gl_start_col, panel);
    panel->len = psz.len_lbcast;

    if(panel->max_lwork_size < (size_t)(psz.lwork) * sizeof(double)) {
      //if(panel->LWORK) {
      //  free(panel->LWORK);
      //}
      // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) +
      // (size_t)4095)/(size_t)4096)*(size_t)4096;
      const size_t numbytes = (size_t)(psz.lwork) * sizeof(double);

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

      panel->max_lwork_size = (size_t)(psz.lwork) * sizeof(double);
    }
    if(panel->max_uwork_size < (size_t)(psz.uwork) * sizeof(double)) {
      //if(panel->UWORK) {
      //  free(panel->UWORK);
      //}
      // size_t numbytes = (((size_t)((size_t)(uwork) * sizeof( double )) +
      // (size_t)4095)/(size_t)4096)*(size_t)4096;
      const size_t numbytes = (size_t)(psz.uwork) * sizeof(double);

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

      panel->max_uwork_size = (size_t)(psz.uwork) * sizeof(double);
    }

    /*
     * Initialize the pointers of the panel structure
     */
    initialize_panel_pointers(mat, ncols, icurrow, icurcol, mp, psz.ml2, psz.nu, psz.lpiv, fraction,
                              panel);

    *(panel->DINFO) = 0.0;

    //const int liwork = get_index_workspace_len(nprow, ncols, mp);

    if(panel->max_iwork_size < psz.l_i_work * sizeof(int)) {
      const size_t numbytes = psz.l_i_work * sizeof(int);

      if(HPL_malloc((void**)&(panel->IWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for integer workspace.");
      }
      panel->max_iwork_size = psz.l_i_work * sizeof(int);
    }

    if(psz.l_i_work) {
      *(panel->IWORK) = -1;
    }

    /*Finally, we need 4 + 4*ncols entries of scratch for pdfact */
    //const auto lfwork = static_cast<size_t>(((4 + ((unsigned int)(ncols) << 1)) << 1));
    if(panel->max_fwork_size < psz.l_f_work * sizeof(double)) {
      const size_t numbytes = psz.l_f_work * sizeof(double);

      if(HPL_malloc((void**)&(panel->fWORK), numbytes) != HPL_SUCCESS) {
        HPL_pabort(__LINE__,
                   "HPL_pdpanel_init",
                   "Host memory allocation failed for pdfact scratch workspace.");
      }
      panel->max_fwork_size = psz.l_f_work * sizeof(double);
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
