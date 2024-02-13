#include "test_config.hpp"

#include <string>

#include <hpl_pfact.hpp>

namespace test {

// test_config::test_config()
// {
//     nrows = 1;
//     block_size = 1;
//     nprows = 1;
//     npcols = 1;
//     algo = get_default_settings();
// }

// test_config::test_config(const int argc, char *argv[])
// {
//     nrows = -1;
//     block_size = -1;
//     nprows = 1;
//     npcols = 1;
//     algo = get_default_settings();
//     for(int i = 1; i < argc; i++) {
//         const std::string option = argv[i];
//         if(option == "-N" || option == "--nrows") {
//             nrows = std::stoi(argv[i+1]);
//             i++;
//         }
//         else if(option == "-b" || option == "--block_size") {
//             block_size = std::stoi(argv[i+1]);
//             i++;
//         }
//         else if(option == "-P" || option == "--n_proc_rows") {
//             nprows = std::stoi(argv[i+1]);
//             i++;
//         }
//         else if(option == "-Q" || option == "--n_proc_cols") {
//             npcols = std::stoi(argv[i+1]);
//             i++;
//         }
//     }

//     if(nrows == -1 || block_size == -1) {
//         throw std::runtime_error("Need at least matrix dimension and block size!");
//     }
// }

extra_params get_extra_params(const int argc, char *argv[])
{
    extra_params params;
    params.nrepeats = 20;

    for(int i = 1; i < argc; i++) {
        const std::string option = argv[i];
        if(option == "--nrepeats") {
            params.nrepeats = std::stoi(argv[i+1]);
            i++;
        }
    }
    return params;
}

HPL_T_palg get_default_settings()
{
    HPL_T_palg algo;
    algo.btopo = HPL_1RING;
    algo.comm_impls_types = {HPL_COMM_CUSTOM_IMPL, HPL_COMM_CUSTOM_IMPL,
                             HPL_COMM_CUSTOM_IMPL, HPL_COMM_CUSTOM_IMPL};
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
    algo.frac = 0.6;
    return algo;
}

HPL_T_palg get_algo_from_test(const HPL_Test_params &params)
{
    HPL_T_palg algo;
    algo.btopo = params.bcast_algos[0];
    algo.depth = params.lookahead_depths[0];
    algo.nbmin = params.recursive_stop_crit[0];
    algo.nbdiv = params.num_panels_recursion[0];
    algo.comm_impls_types.bcast_type = params.bcast_type;
    algo.comm_impls_types.allreduce_dmxswp_type = params.allreduce_dmxswp_type;
    algo.comm_impls_types.allgatherv_type = params.allgatherv_type;
    algo.comm_impls_types.scatterv_type = params.scatterv_type;

    algo.pfact = params.panel_facts[0];

    if(params.L1_no_transpose) {
        if(params.panel_facts[0] == HPL_LEFT_LOOKING)
            algo.pffun = HPL_pdpanllN;
        else if(params.panel_facts[0] == HPL_CROUT)
            algo.pffun = HPL_pdpancrN;
        else
            algo.pffun = HPL_pdpanrlN;

        algo.rfact = params.recursive_facts[0];
        if(params.recursive_facts[0] == HPL_LEFT_LOOKING)
            algo.rffun = HPL_pdrpanllN;
        else if(params.recursive_facts[0] == HPL_CROUT)
            algo.rffun = HPL_pdrpancrN;
        else
            algo.rffun = HPL_pdrpanrlN;

        //algo.upfun = HPL_pdupdateNT;
    } else {
        if(params.panel_facts[0] == HPL_LEFT_LOOKING)
            algo.pffun = HPL_pdpanllT;
        else if(params.panel_facts[0] == HPL_CROUT)
            algo.pffun = HPL_pdpancrT;
        else
            algo.pffun = HPL_pdpanrlT;

        algo.rfact = params.recursive_facts[0];
        if(params.recursive_facts[0] == HPL_LEFT_LOOKING)
            algo.rffun = HPL_pdrpanllT;
        else if(params.recursive_facts[0] == HPL_CROUT)
            algo.rffun = HPL_pdrpancrT;
        else
            algo.rffun = HPL_pdrpanrlT;

        //algo.upfun = HPL_pdupdateTT;
    }

    algo.fswap = params.fswap;
    algo.fsthr = params.swap_threshold_cols;
    algo.equil = params.equil;
    algo.align = params.mem_align;

    algo.frac = params.frac;

    return algo;
}

}
