#ifndef HPL_TEST_MATGEN_HPP_
#define HPL_TEST_MATGEN_HPP_

namespace test {
namespace matgen {

/// Generates a matrix on one process on main memory
void HPL_dmatgen(int M, int N, double *A, int LDA, int ISEED);

constexpr int HPL_MULT0 =       1284865837;
constexpr int HPL_MULT1 =       1481765933;
constexpr int HPL_IADD0 =       1;
constexpr int HPL_IADD1 =       0;
constexpr double HPL_DIVFAC = 2147483648.0;
constexpr double HPL_POW16  = 65536.0;
constexpr double HPL_HALF   = 0.5;

double HPL_rand();
void HPL_xjumpm(int JUMPM, int *MULT, int *IADD, int *IRANN, int *IRANM, int *IAM, int *ICM);
void HPL_jumpit(int *MULT, int *IADD, int *IRANN, int *IRANM);
void HPL_lmul(int *K, int *J, int *I);
void HPL_ladd(int *J, int *K, int *I);

void HPL_setran(const int OPTION, int *const IRAN);

}
}

#endif
