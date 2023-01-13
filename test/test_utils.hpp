#ifndef ORNL_HPL_TEST_DATA_H_
#define ORNL_HPL_TEST_DATA_H_

#include <string>

#include "hpl_host.hpp"
#include "hpl_ptest.hpp"

namespace ornl_hpl {
namespace test {

struct TestData {
    int global_size;
    int block_size;
    std::string reference_solution_path;
};

TestData initialize_grid_and_test(int argc, char *argv[], HPL_T_grid *grid, HPL_T_test *test);

}
}

#endif
