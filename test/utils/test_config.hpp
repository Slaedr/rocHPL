#ifndef HPL_TEST_CONFIG_HPP_
#define HPL_TEST_CONFIG_HPP_

#include <hpl_pgesv_types.hpp>
#include <hpl_ptest.hpp>

namespace test {

HPL_T_palg get_default_settings();

HPL_T_palg get_algo_from_test(const HPL_Test_params &params);

}


#endif // HPL_TEST_CONFIG_HPP_
