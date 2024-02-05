#include <iostream>
#include <iomanip>
#include <omp.h>

#include <hpl_pfact.hpp>

#include "../matgen/matgen.hpp"
#include "../matgen/alloc.hpp"

using namespace test;

HPL_T_palg get_default_settings();

void test_shared_pdrpanrlN(HPL_T_panel *const panel)
{
    HPL_pdfact(panel, HPL_COMM_CUSTOM_IMPL);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const int nrows = 256;
    const int bs = 64;
    const int ncols = bs;
    const int lda = nrows;
    const int seed = 46;

    HPL_T_grid grid;
    HPL_grid_init(MPI_COMM_WORLD, HPL_COLUMN_MAJOR, 1, 1, 1, 1, &grid);
    const HPL_T_palg algo = get_default_settings();

    HPL_T_pmat mat;
    HPL_host_pdmat_init(&grid, nrows, ncols, &mat);
    matgen::HPL_dmatgen(nrows, ncols, mat.A, lda, seed);

    HPL_T_panel panel;
    allocate_host_panel(&grid, &algo, &mat, nrows, ncols, ncols, 0, 0, &panel);

    test_shared_pdrpanrlN(&panel);

    free_host_panel(&panel);
    HPL_host_matfree(&mat);
    HPL_grid_exit(&grid);
    MPI_Finalize();
    return 0;
}

HPL_T_palg get_default_settings()
{
    HPL_T_palg algo;
    algo.depth = 1;
    algo.nbdiv = 2;
    algo.nbmin = 16;
    algo.pfact = HPL_RIGHT_LOOKING;
    algo.rfact = HPL_RIGHT_LOOKING;
    algo.pffun = HPL_pdpanrlN;
    algo.rffun = HPL_pdrpanrlN;
    algo.fswap = HPL_SWAP01;
    algo.fsthr = 64;
    algo.equil = 0;
    algo.align = 8;
    return algo;
}
