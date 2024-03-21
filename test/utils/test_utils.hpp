#ifndef ROCHPL_TEST_UTILS_MAT_UTILS_HPP_
#define ROCHPL_TEST_UTILS_MAT_UTILS_HPP_

#include <vector>

namespace test {

struct comparison {
    bool equal;
    int diff_position;
};

template <typename T>
comparison test_equality(const std::vector<T>& A, const std::vector<T>& B, const double rel_tol);

}

#endif
