
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
    if(thread_size == 1) {
        HPL_dtrsm(order, side, uplo, trans, diag,    
                  M, N, ALPHA,    
                  A, LDA, B, LDB);
        return;
    }

    if(thread_rank == 0) {
        return;
    }

    for(int j = 0, tile = 0; j < N; j += t_item_size, ++tile) {    
        if(tile % (thread_size-1) == (thread_rank-1)) {
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
