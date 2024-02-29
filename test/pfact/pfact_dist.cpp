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

void pfact_dist(const HPL_T_grid *const grid, const HPL_Test_params& tparams, const HPL_T_test& test,
                const extra_params& eparams)
{
    const int bs = tparams.bs[0];

    HPL_T_pmat mat;
    HPL_host_pdmat_init(grid, tparams.matrix_sizes[0], eparams.n_gl_cols, bs, &mat);
    const int seed = 46;

    MPI_Type_contiguous(bs + 4, MPI_DOUBLE, &PDFACT_ROW);
    MPI_Type_commit(&PDFACT_ROW);


    const HPL_T_palg algo = get_algo_from_test(tparams);


    double wtime;
    double HPL_w[HPL_TIMING_N];
    HPL_ptimer_boot();
    HPL_ptimer(0);

    if(eparams.n_gl_cols % bs != 0) {
        throw PreconditionNotMet("Num cols not divisible by block size!");
    }
    const int n_block_cols = eparams.n_gl_cols / bs;

    int num_pdfacts = 0;
  
    for(int irpt = 0; irpt < eparams.nrepeats; irpt++) {
        if(grid->iam == 0) {
            std::cout << "Repetition " << irpt << std::endl;
        }
        generate_random_values_host(&mat, seed);

        for(int iblk = 0; iblk < n_block_cols; iblk++) {
            HPL_T_panel panel;
            allocate_host_panel(grid, &algo, &mat, tparams.matrix_sizes[0] - iblk*bs,
                                eparams.n_gl_cols - iblk*bs + 1, bs,
                                iblk*tparams.bs[0], iblk*tparams.bs[0], &panel);
            //allocate_host_panel(grid, &algo, &mat, tparams.matrix_sizes[0],
            //                    eparams.n_gl_cols + 1, bs, 0, 0, &panel);

            HPL_pdfact(&panel, HPL_COMM_CUSTOM_IMPL);
            num_pdfacts++;

            free_host_panel(&panel);

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

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
        std::cout << "Num pfacts = " << num_pdfacts << std::endl;
        std::fprintf(test.outfp, "Num test repeats = %d.\n\n", eparams.nrepeats);
        std::fprintf(test.outfp,
                      "%s%s\n",
                      "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                      "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-");
        /*
         * Recursive panel factorization
         */
        const double t_recfact = HPL_w[HPL_TIMING_RPFACT - HPL_TIMING_BEG];
        if(t_recfact > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ Max aggregated wall time rfact . . : %18.2f\n", t_recfact);
        }
        /*
         * Panel factorization
         */
        const double t_basefact = HPL_w[HPL_TIMING_PFACT - HPL_TIMING_BEG];
        if(t_basefact > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ + Max aggregated wall time pfact . : %18.2f\n", t_basefact);
        }
        /*
         * Panel factorization (swap)
         */
        const double t_mxswp = HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG];
        if(t_mxswp > HPL_rzero) {
            std::fprintf(test.outfp,
                        "+ + Max aggregated wall time mxswp . : %18.2f\n", t_mxswp);
        }

        std::fprintf(test.outfp, "Exclusive times:\n");
        std::fprintf(test.outfp, "Recursive fact       : %18.2f\n", t_recfact-t_basefact);
        std::fprintf(test.outfp, "Base fact            : %18.2f\n", t_basefact-t_mxswp);
        std::fprintf(test.outfp, "Fact max swap        : %18.2f\n", t_mxswp);
    }

    MPI_Type_free(&PDFACT_ROW);
    HPL_host_matfree(&mat);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Op_create(HPL_dmxswp, true, &HPL_DMXSWP);

    HPL_T_test tt;
    HPL_Test_params tparams = HPL_pdinfo(argc, argv, &tt);
    const extra_params eparams = get_extra_params(argc, argv, &tparams);

    HPL_T_grid grid;
    HPL_grid_init(MPI_COMM_WORLD,tparams.process_ordering, tparams.gl_proc_rows[0],tparams.gl_proc_cols[0],
                  tparams.loc_proc_rows, tparams.loc_proc_cols, &grid);

    pfact_dist(&grid, tparams, tt, eparams);

    HPL_grid_exit(&grid);

    MPI_Finalize();
    return 0;
}
