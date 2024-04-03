#include "test_utils.hpp"

#include <algorithm>

namespace test {

template <typename T>
comparison test_equality(const std::vector<T>& A, const std::vector<T>& B, const double rel_tol)
{
    comparison res{true};
    if(A.size() != B.size()) {
        res.equal = false;
        return res;
    }

    for(int i = 0; i < static_cast<int>(A.size()); i++) {
        const T diff = std::abs(A[i] - B[i]);
        const T base = std::max(std::abs(A[i]), std::abs(B[i]));
        if(diff/base > rel_tol) {
            res.equal = false;
            res.diff_position = i;
            break;
        }
    }

    return res;
}

template comparison test_equality(const std::vector<double>& A, const std::vector<double>& B,
        const double rel_tol);

}
