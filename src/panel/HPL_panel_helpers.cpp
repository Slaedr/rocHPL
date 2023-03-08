#include "hpl_panel.hpp"
#include "hpl_pauxil.hpp"

int get_num_rows_L2(const int npcol, const int myrow, const int current_prow,
        const int panel_ncols, const int local_nrows)
{
    int ml2 = 0;
    if(npcol > 1) {
        ml2 = (myrow == current_prow ? local_nrows - panel_ncols : local_nrows);
        ml2 = Mmax(0, ml2);
        ml2 = ((ml2 + 95) / 128) * 128 + 32; /*pad*/
    } else {
        ml2 = 0; // L2 is aliased inside A
    }
    return ml2;
}

int get_index_workspace_len(const int nprow, const int panel_ncols, const int local_nrows)
{
    /*
     * If nprow is 1, we just allocate an array of JB (panel_ncols) integers to store the
     * pivot IDs during factoring, and a scratch array of mp integers.
     * When nprow > 1, we allocate the space for the index arrays immediate-
     * ly. The exact size of this array depends on the swapping routine that
     * will be used, so we allocate the maximum:
     *
     *    IWORK[0] is of size at most 1      +
     *    IPL      is of size at most 1      +
     *    IPID     is of size at most 4 * JB +
     *    IPIV     is of size at most JB     +
     *    SCRATCH  is of size at most MP
     *
     *    ipA      is of size at most 1      +
     *    iplen    is of size at most NPROW  + 1 +
     *    ipcounts is of size at most NPROW  +
     *    ioffsets is of size at most NPROW  +
     *    iwork    is of size at most MAX( 2*JB, NPROW+1 ).
     *
     * that is  mp + 4 + 5*JB + 3*NPROW + MAX( 2*JB, NPROW+1 ).
     *
     * We use the fist entry of this to work array  to indicate  whether the
     * the  local  index arrays have already been computed,  and if yes,  by
     * which function:
     *    IWORK[0] = -1: no index arrays have been computed so far;
     *    IWORK[0] =  1: HPL_pdlaswp already computed those arrays;
     * This allows to save some redundant and useless computations.
     */
    int lwork = 0;
    if(nprow == 1) {
        lwork = local_nrows + panel_ncols;
    } else {
        int itmp1 = panel_ncols * 2;
        lwork = nprow + 1;
        itmp1 = Mmax(itmp1, lwork);
        lwork = local_nrows + 4 + (5 * panel_ncols) + (3 * nprow) + itmp1;
    }
    return lwork;
}

HPL_panel_sizes get_panel_sizes(HPL_T_pmat *A, const int M, const int N, const int JB,
                                const int IA, const int JA,
                                HPL_T_panel *panel)
{
    HPL_panel_sizes psz;
    int icurcol, icurrow, ii, jj;
    const int myrow = panel->grid->myrow;
    const int mycol = panel->grid->mycol;
    const int nprow = panel->grid->nprow;
    const int npcol = panel->grid->npcol;
    const int nb    = A->nb;
    HPL_infog2l(IA,
              JA,
              nb,
              nb,
              nb,
              nb,
              0,
              0,
              myrow,
              mycol,
              nprow,
              npcol,
              &ii,
              &jj,
              &icurrow,
              &icurcol);
    const int mp = HPL_numrocI(M, IA, nb, nb, myrow, 0, nprow);
    const int nq = HPL_numrocI(N, JA, nb, nb, mycol, 0, npcol);

    const int inxtcol = MModAdd1(icurcol, npcol);
    const int inxtrow = MModAdd1(icurrow, nprow);

    psz.lpiv = (5 * JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));
    psz.ml2 = get_num_rows_L2(npcol, myrow, icurrow, JB, mp);
    psz.len_lbcast = psz.ml2 * JB + JB * JB + psz.lpiv; // L2, L1, integer arrays
    psz.lwork = psz.len_lbcast + 1;
    psz.nu  = Mmax(0, (mycol == icurcol ? nq - JB : nq));
    psz.ldu = psz.nu + JB + 256; /*extra space for potential padding*/
    psz.uwork = JB * psz.ldu;
    psz.l_i_work = static_cast<size_t>(get_index_workspace_len(nprow, JB, mp));
    psz.l_f_work = static_cast<size_t>((4 + ((unsigned int)(JB) << 1)) << 2);

    return psz;
}

/*
 * Initialize the pointers of the panel structure
 */
void initialize_panel_pointers(HPL_T_pmat *const A,
                               const int JB, const int icurrow, const int icurcol, const int mp,
                               const int ml2, const int nu, const int lpiv,
                               const double fraction,
                               HPL_T_panel *const panel)
{
  const int myrow = panel->grid->myrow;
  const int mycol = panel->grid->mycol;
  const int nprow = panel->grid->nprow;
  const int npcol = panel->grid->npcol;
  const int inxtcol = MModAdd1(icurcol, npcol);
  const int inxtrow = MModAdd1(icurrow, nprow);
  const int nb = A->nb;

  if(npcol == 1) {
    panel->L2    = panel->A + (myrow == icurrow ? JB : 0);
    panel->dL2   = panel->dA + (myrow == icurrow ? JB : 0);
    panel->ldl2  = A->ld;
    panel->dldl2 = A->ld; /*L2 is aliased inside A*/

    panel->L1  = (double*)panel->LWORK;
    panel->dL1 = (double*)panel->dLWORK;
  } else {
    panel->L2    = (double*)panel->LWORK;
    panel->dL2   = (double*)panel->dLWORK;
    panel->ldl2  = Mmax(0, ml2);
    panel->dldl2 = Mmax(0, ml2);

    panel->L1  = panel->L2 + ml2 * JB;
    panel->dL1 = panel->dL2 + ml2 * JB;
  }

  panel->U  = (double*)panel->UWORK;
  panel->dU = (double*)panel->dUWORK;
  panel->W  = A->W;
  panel->dW = A->dW;

  if(nprow == 1) {
    panel->nu0  = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    panel->ldu0 = panel->nu0;

    panel->nu1  = 0;
    panel->ldu1 = 0;

    panel->nu2  = nu - panel->nu0;
    panel->ldu2 = ((panel->nu2 + 95) / 128) * 128 + 32; /*pad*/

    panel->U2  = panel->U + JB * JB;
    panel->dU2 = panel->dU + JB * JB;
    panel->U1  = panel->U2 + panel->ldu2 * JB;
    panel->dU1 = panel->dU2 + panel->ldu2 * JB;

    panel->permU  = (int*)(panel->L1 + JB * JB);
    panel->dpermU = (int*)(panel->dL1 + JB * JB);
    panel->ipiv   = panel->permU + JB;
    panel->dipiv  = panel->dpermU + JB;

    panel->DINFO  = (double*)(panel->ipiv + 2 * JB);
    panel->dDINFO = (double*)(panel->dipiv + 2 * JB);
  } else {
    const int NSplit = Mmax(0, ((((int)(A->nq * fraction)) / nb) * nb));
    panel->nu0       = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    panel->ldu0      = panel->nu0;

    panel->nu2  = Mmin(nu - panel->nu0, NSplit);
    panel->ldu2 = ((panel->nu2 + 95) / 128) * 128 + 32; /*pad*/

    panel->nu1  = nu - panel->nu0 - panel->nu2;
    panel->ldu1 = ((panel->nu1 + 95) / 128) * 128 + 32; /*pad*/

    panel->U2  = panel->U + JB * JB;
    panel->dU2 = panel->dU + JB * JB;
    panel->U1  = panel->U2 + panel->ldu2 * JB;
    panel->dU1 = panel->dU2 + panel->ldu2 * JB;

    panel->W2  = panel->W + JB * JB;
    panel->dW2 = panel->dW + JB * JB;
    panel->W1  = panel->W2 + panel->ldu2 * JB;
    panel->dW1 = panel->dW2 + panel->ldu2 * JB;

    panel->lindxA   = (int*)(panel->L1 + JB * JB);
    panel->dlindxA  = (int*)(panel->dL1 + JB * JB);
    panel->lindxAU  = panel->lindxA + JB;
    panel->dlindxAU = panel->dlindxA + JB;
    panel->lindxU   = panel->lindxAU + JB;
    panel->dlindxU  = panel->dlindxAU + JB;
    panel->permU    = panel->lindxU + JB;
    panel->dpermU   = panel->dlindxU + JB;

    // Put ipiv array at the end
    panel->ipiv  = panel->permU + JB;
    panel->dipiv = panel->dpermU + JB;

    panel->DINFO  = ((double*)panel->lindxA) + lpiv;
    panel->dDINFO = ((double*)panel->dlindxA) + lpiv;
  }
}
