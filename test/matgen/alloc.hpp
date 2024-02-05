#ifndef HPL_TEST_ALLOC_HPP_
#define HPL_TEST_ALLOC_HPP_

#include <hpl_grid.hpp>
#include <hpl_pgesv_types.hpp>
#include <hpl_panel.hpp>

namespace test {

int HPL_host_pdmat_init(const HPL_T_grid* GRID, int N, int NB, HPL_T_pmat *mat);

void HPL_host_matfree(HPL_T_pmat *mat);

void allocate_host_panel(const HPL_T_grid *grid, const HPL_T_palg *algo, HPL_T_pmat *mat,
                         int nrows, int gl_trailing_ncols, int ncols,
                         int gl_start_row, int gl_start_col,
                         HPL_T_panel *panel);

void free_host_panel(HPL_T_panel *panel);

}

#endif // HPL_TEST_ALLOC_HPP_
