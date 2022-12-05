#ifndef ROCHPL_MAT_HPP_
#define ROCHPL_MAT_HPP_

typedef struct HPL_S_pmat {
  double* dA;   /* pointer to local piece of A */
  double* dX;   /* pointer to solution vector */
  int     n;    /* global problem size */
  int     nb;   /* blocking factor */
  int     ld;   /* local leading dimension */
  int     mp;   /* local number of rows */
  int     nq;   /* local number of columns */
  int     info; /* computational flag */
  double* A;
  double* W;
  double* dW;
} HPL_T_pmat;

#endif
