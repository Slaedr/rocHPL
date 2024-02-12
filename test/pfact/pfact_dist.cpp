#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>

#include <hpl_pfact.hpp>
#include <hpl_ptimer.hpp>
#include <hpl_ptest.hpp>

#include "../matgen/matrix.hpp"
#include "../matgen/panel.hpp"
#include "../utils/error_handling.hpp"
#include "../utils/test_config.hpp"

using namespace test;

constexpr double reltol = 1e-10;

void pfact_dist(const HPL_T_grid *const grid, const HPL_Test_params& tparams, const HPL_T_test& test)
{
    HPL_T_pmat mat;
    HPL_host_pdmat_init(grid, tparams.matrix_sizes[0], tparams.bs[0], &mat);
    const int seed = 46;
    generate_random_values_host(&mat, seed);

    MPI_Type_contiguous(tparams.bs[0] + 4, MPI_DOUBLE, &PDFACT_ROW);
    MPI_Type_commit(&PDFACT_ROW);


    HPL_T_panel panel;
    const HPL_T_palg algo = get_algo_from_test(tparams);
    allocate_host_panel(grid, &algo, &mat, tparams.matrix_sizes[0],
                        tparams.matrix_sizes[0]+1, tparams.bs[0],
                        tparams.loc_proc_rows, tparams.loc_proc_cols, &panel);

    double wtime;
    double HPL_w[HPL_TIMING_N];
    HPL_ptimer_boot();
    HPL_ptimer(0);
    HPL_pdfact(&panel, HPL_COMM_CUSTOM_IMPL);
    HPL_ptimer(0);
    HPL_ptimer_combine(
        grid->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME, 1, 0, &wtime);

    HPL_ptimer_combine(grid->all_comm,
                       HPL_AMAX_PTIME,
                       HPL_WALL_PTIME,
                       HPL_TIMING_N,
                       HPL_TIMING_BEG,
                       HPL_w);
    if((grid->myrow == 0) && (grid->mycol == 0)) {
        std::fprintf(test.outfp,
                      "%s%s\n",
                      "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                      "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-");
        /*
         * Recursive panel factorization
         */
        if(HPL_w[HPL_TIMING_RPFACT - HPL_TIMING_BEG] > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ Max aggregated wall time rfact . . : %18.2f\n",
                        HPL_w[HPL_TIMING_RPFACT - HPL_TIMING_BEG]);
        }
        /*
         * Panel factorization
         */
        if(HPL_w[HPL_TIMING_PFACT - HPL_TIMING_BEG] > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ + Max aggregated wall time pfact . : %18.2f\n",
                        HPL_w[HPL_TIMING_PFACT - HPL_TIMING_BEG]);
        }
        /*
         * Panel factorization (swap)
         */
        if(HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG] > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ + Max aggregated wall time mxswp . : %18.2f\n",
                        HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG]);
        }
    }

    MPI_Type_free(&PDFACT_ROW);
    free_host_panel(&panel);
    HPL_host_matfree(&mat);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Op_create(HPL_dmxswp, true, &HPL_DMXSWP);

    HPL_T_test tt;
    HPL_Test_params tparams = HPL_pdinfo(argc, argv, &tt);

    for(int itest = 0; itest < tparams.matrix_sizes.size(); itest++) {
        HPL_T_grid grid;
        HPL_grid_init(MPI_COMM_WORLD, tparams.process_ordering, tparams.gl_proc_rows[0], tparams.gl_proc_cols[0],
                      tparams.loc_proc_rows, tparams.loc_proc_cols, &grid);

        pfact_dist(&grid, tparams, tt);

        HPL_grid_exit(&grid);
    }
    MPI_Finalize();
    return 0;
}
