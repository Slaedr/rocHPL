#ifndef ROCHPL_GATHER_MATRIX_HPP_
#define ROCHPL_GATHER_MATRIX_HPP_

#include <vector>
#include <string>

#include "hpl_host.hpp"

namespace ornl_hpl {
namespace test {

/**
 * Gathers a distributed matrix on rank 0 and returns a device pointer.
 */
std::vector<double> gather_matrix(const HPL_T_grid *const grid, const HPL_T_pmat *const mat);

/**
 * Reads a matrix sequentially from a single file.
 */
std::vector<double> read_matrix_single_block(const std::string& path, const int global_size);

/**
 * Checks for equality (pointwise, relative) of two buffers to within 10 times the machine epsilon.
 */
bool compare_gathered_matrices(const std::vector<double>& A, const std::vector<double>& b, const int num_rows);

}
}

#endif
