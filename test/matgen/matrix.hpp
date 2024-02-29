#ifndef HPL_TEST_MATRIX_HPP_
#define HPL_TEST_MATRIX_HPP_

#include <hpl_grid.hpp>
#include <hpl_pgesv_types.hpp>

namespace test {

int HPL_host_pdmat_init(const HPL_T_grid* GRID, int nrows, int ncols, int NB, HPL_T_pmat *mat);

void HPL_host_matfree(HPL_T_pmat *mat);

/**
 * Generates random values in the top left mp x nb local matrix.
 */
void generate_random_values_host(HPL_T_pmat *const mat, const int seed);

/// Difference between two matrices
struct mat_diff {
    bool match_mp;        ///< Are the local num. rows. the same?
    bool match_nb;        ///< Are the block sizes the same?
    int iA;               ///< Row of first difference
    int jA;               ///< Col of first difference
    double rel_diff_norm; ///< Relative max norm of difference
};

/// Compute the relative max-norm difference and the first position of difference
mat_diff compare_matrices_host(const HPL_T_pmat *m1, const HPL_T_pmat *m2, double reltol);

/**
 * \brief Checks if two matrices are significantly different compared to a threshold.
 *
 * Prints rel norm difference.
 * Throws PreconditionNotMet if there is significant difference.
 */
void test_mat_same_host(const HPL_T_pmat *m1, const HPL_T_pmat *m2, double reltol);

}

#endif // HPL_TEST_ALLOC_HPP_
