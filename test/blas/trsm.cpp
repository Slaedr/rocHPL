#include <vector>
#include <string>
#include <omp.h>

#include <hpl_blas.hpp>

#include "../matgen/matgen.hpp"
#include "../utils/test_utils.hpp"
#include "../utils/error_handling.hpp"

using namespace test;

int main(int argc, char *argv[])
{
    const int seed = 41;
    const int t_item_size = 5;
    const int m = 5, n = 16;
    const int lda = 10, ldb = 11;
    std::vector<double> A(m*lda);
    std::vector<double> B(n*ldb);
    matgen::HPL_dmatgen(m, m, A.data(), lda, seed);
    matgen::HPL_dmatgen(m, n, B.data(), ldb, seed);
    std::vector<double> A_test = A;
    std::vector<double> B_test = B;

    cblas_dtrsm(HplColumnMajor, HplLeft, HplLower, HplNoTrans, HplUnit,
            m, n, 1.0, A.data(), lda, B.data(), ldb);
        
#pragma omp parallel
    {
        const int thread_rank = omp_get_thread_num();
        const int thread_size = omp_get_num_threads();
        HPL_dtrsm_omp(HplColumnMajor, HplLeft, HplLower, HplNoTrans, HplUnit,
                m, n, 1.0, A_test.data(), lda, B_test.data(), ldb,
                t_item_size, thread_rank, thread_size);
    }

    // Compare
    const auto co = test_equality(B, B_test, 1e-14);

    if(!co.equal) {
        throw TestFailed("Matrices not equal at " + std::to_string(co.diff_position));
    }

    return 0;
}
