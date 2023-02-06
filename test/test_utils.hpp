#ifndef ORNL_HPL_TEST_DATA_H_
#define ORNL_HPL_TEST_DATA_H_

#include <string>
#include <stdexcept>

#include "hpl_host.hpp"
#include "hpl_ptest.hpp"

namespace ornl_hpl {
namespace test {

class SeqTestFailed : public std::runtime_error
{
public:
    SeqTestFailed(const std::string file, const int line)
        : std::runtime_error("Test failed at " + file + ": " + std::to_string(line))
    { }
};

#define HPL_ASSERT_THROW_SEQ(_val) \
    if(!(_val)) { \
        throw ornl_hpl::test::SeqTestFailed(__FILE__, __LINE__); \
    } \
    static_assert(true, "dummy")

struct TestData {
    int global_size;
    int block_size;
    std::string reference_solution_path;
};

TestData initialize_grid_and_test(int argc, char *argv[], HPL_T_grid *grid, HPL_T_test *test);

}
}

#endif
