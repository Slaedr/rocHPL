#include <iostream>
#include <iomanip>
#include <omp.h>

#include <hpl_pfact.hpp>

#include "../matgen/matgen.hpp"

using namespace test;

int main(int argc, char *argv[])
{
    const int nrows = 256;
    const int bs = 64;
    const int ncols = bs;
    const int lda = nrows;
    const int seed = 46;

    auto matrix = new double[nrows*ncols];
    matgen::HPL_dmatgen(nrows, ncols, matrix, lda, seed);

    std::cout << "Entry at 14,13 = " << matrix[14 + 13*lda] << std::endl;

    double max_value;
    int max_index;
    HPL_T_panel panel;

#pragma omp parallel
    {
        const int thread_rank = omp_get_thread_num();
        const int thread_size = omp_get_num_threads();
        HPL_pdrpanrlN(&panel, 0, 0, 0, static_cast<double*>(nullptr),
                thread_rank, thread_size,
                &max_value, &max_index, HPL_COMM_COLLECTIVE);
    }
    delete [] matrix;
    return 0;
}
