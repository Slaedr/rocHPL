#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>

#include <hpl_pfact.hpp>

#include "../matgen/matrix.hpp"
#include "../matgen/panel.hpp"
#include "../utils/error_handling.hpp"

using namespace test;

HPL_T_palg get_default_settings();
constexpr double reltol = 1e-10;

void test_shared_pdrpanrlN(HPL_T_panel *const panel_1, HPL_T_panel *const panel_2)
{
    HPL_pdfact(panel_1, HPL_COMM_CUSTOM_IMPL);
    HPL_pdfact(panel_2, HPL_COMM_CUSTOM_IMPL);
    const auto diff = compare_panels_host(panel_1, panel_2, reltol);
    if(diff.match_M && diff.match_ncols) {
        std::cout << "Rel norm diff in A: " << diff.rel_diff_norm_A << std::endl;
        std::cout << "Rel norm diff in L1: " << diff.rel_diff_norm_L1 << std::endl;
        if(diff.i_A != -1) {
            throw TestFailed("panels differ in A at " + std::to_string(diff.i_A) + "," + std::to_string(diff.j_A));
        }
        if(diff.i_L1 != -1) {
            throw TestFailed("panels differ in L1 at " + std::to_string(diff.i_L1) + "," + std::to_string(diff.j_L1));
        }
    } else {
        throw TestFailed("Panels differ in sizes!");
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const int nrows = 256;
    const int bs = 64;
    const int ncols = bs;
    const int seed = 46;

    HPL_T_grid grid;
    HPL_grid_init(MPI_COMM_WORLD, HPL_COLUMN_MAJOR, 1, 1, 1, 1, &grid);
    const HPL_T_palg algo1 = get_default_settings();
    HPL_T_palg algo2 = get_default_settings();
    algo2.rffun = HPL_pdrpanrlN;
    algo2.pffun = HPL_pdpanrlN;

    HPL_T_pmat mat1, mat2;
    HPL_host_pdmat_init(&grid, nrows, nrows, bs, &mat1);
    generate_random_values_host(&mat1, seed);
    HPL_host_pdmat_init(&grid, nrows, nrows, bs, &mat2);
    generate_random_values_host(&mat2, seed);
    test_mat_same_host(&mat1, &mat2, 2.2e-16);

    HPL_T_panel panel1, panel2;
    allocate_host_panel(&grid, &algo1, &mat1, nrows, ncols, bs, 0, 0, &panel1);
    allocate_host_panel(&grid, &algo2, &mat2, nrows, ncols, bs, 0, 0, &panel2);

    test_shared_pdrpanrlN(&panel1, &panel2);

    free_host_panel(&panel1);
    free_host_panel(&panel2);
    HPL_host_matfree(&mat1);
    HPL_host_matfree(&mat2);
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
