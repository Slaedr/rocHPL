#ifndef HPL_TEST_CONFIG_HPP_
#define HPL_TEST_CONFIG_HPP_

#include <hpl_pgesv_types.hpp>
#include <hpl_ptest.hpp>

namespace test {

struct extra_params {
    int nrepeats;
    int n_gl_cols;
};

HPL_T_palg get_default_settings();

HPL_T_palg get_algo_from_test(const HPL_Test_params &params);

extra_params get_extra_params(int argc, char *argv[], const HPL_Test_params *tparams);

}


#endif // HPL_TEST_CONFIG_HPP_
