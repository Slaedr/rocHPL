#ifndef HPL_TEST_PFACT_DIST_HPP_
#define HPL_TEST_PFACT_DIST_HPP_

#include <hpl_pfact.hpp>

namespace test {
namespace pfact {

HPL_T_palg get_cmd_settings(const int argc, const char *argv[]);

HPL_T_palg get_default_settings();

void pfact_dist(HPL_T_panel *const panel);

}
}


#endif // HPL_TEST_PFACT_DIST_HPP_
