
#include "hpl.hpp"

void HPL_dtrsm_omp(const HPL_ORDER order,
                   const enum HPL_SIDE side,
                   const HPL_UPLO uplo,
                   const enum HPL_TRANS trans,
                   const HPL_DIAG diag,
                   const int            M,
                   const int            N,
                   const double         ALPHA,
                   const double* __restrict__ A,
                   const int            LDA,
                   double* __restrict__ B,
                   const int            LDB,
                   const int            t_item_size,
                   const int            thread_rank,
                   const int            thread_size)
{
    for(int j = 0, tile = 0; j < N; j += t_item_size, ++tile) {
        if(tile % thread_size == thread_rank) {
            const int nn = Mmin(t_item_size, N - j);
            HPL_dtrsm(order, side, uplo, trans, diag,
                      M,
                      nn,
                      ALPHA,
                      A,
                      LDA,
                      B + j*LDB,
                      LDB);
        }
    }
}
