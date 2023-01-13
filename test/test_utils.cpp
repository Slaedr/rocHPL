#include "test_utils.hpp"

#include <stdexcept>

namespace ornl_hpl {
namespace test {

TestData initialize_grid_and_test(const int argc, char *argv[], HPL_T_grid *const grid,
                                  HPL_T_test *const test)
{
    int nval[HPL_MAX_PARAM], nbval[HPL_MAX_PARAM], pval[HPL_MAX_PARAM],
        qval[HPL_MAX_PARAM], nbmval[HPL_MAX_PARAM], ndvval[HPL_MAX_PARAM],
        ndhval[HPL_MAX_PARAM];

    HPL_T_FACT pfaval[HPL_MAX_PARAM], rfaval[HPL_MAX_PARAM];

    HPL_T_TOP topval[HPL_MAX_PARAM];

    //HPL_T_palg algo;
    int L1notran, Unotran, align, equil, in, inb, inbm, indh, indv, ipfa, ipq,
        irfa, itop, mycol, myrow, ns, nbs, nbms, ndhs, ndvs, npcol, npfs, npqs,
        nprow, nrfs, ntps, tswap;
    HPL_T_ORDER pmapping;
    HPL_T_FACT  rpfa;
    HPL_T_SWAP  fswap;
    double      frac;
    int         p, q;

    HPL_pdinfo(argc, argv,
               test,
               &ns, nval, &nbs, nbval, &pmapping, &npqs, pval, qval, &p, &q,
               &npfs, pfaval, &nbms, nbmval, &ndvs, ndvval, &nrfs, rfaval,
               &ntps, topval, &ndhs, ndhval,
               &fswap, &tswap,
               &L1notran, &Unotran,
               &equil, &align, &frac);

    std::string ref_mat_path;
    for(int i = 1; i < argc; i++) {
        const std::string cur_opt = argv[i];
        if(cur_opt == "--reference_solution_path") {
            if(argc <= i+1) {
                throw std::runtime_error("Missing cmd line param for test!");
            }
            ref_mat_path = argv[i+1];
        }
    }

    if(npqs < 1 || ns < 1 || nbs < 1) {
        throw std::runtime_error("Not enough command line parameters!");
    }

    HPL_grid_init(MPI_COMM_WORLD, pmapping, pval[0], qval[0], p, q, grid);

    return TestData{nval[0], nbval[0], ref_mat_path};
}

}
}
