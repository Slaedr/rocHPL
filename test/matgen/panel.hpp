#ifndef HPL_TEST_PANEL_H_
#define HPL_TEST_PANEL_H_

#include <hpl_panel.hpp>

namespace test {

void allocate_host_panel(const HPL_T_grid *grid, const HPL_T_palg *algo, HPL_T_pmat *mat,
                         int nrows, int gl_trailing_ncols, int panel_width,
                         int gl_start_row, int gl_start_col,
                         HPL_T_panel *panel);

void free_host_panel(HPL_T_panel *panel);

/// Encodes where two panels have different values
struct hpl_panel_diff {
    bool match_M;         ///< Whether two panels have the same number of global rows
    bool match_ncols;     ///< Whether two panels have the same number of columns
    int i_L1;             ///< row of first differing entry of L1
    int j_L1;             ///< col of first differing entry of L1
    int i_A;
    int j_A;
    double rel_diff_norm_A;
    double rel_diff_norm_L1;
};

/// Determine where two panels differ
hpl_panel_diff compare_panels_host(const HPL_T_panel *p1, const HPL_T_panel *p2, double reltol,
                                   bool transpose_L1_2 = false);

}


#endif // PANEL_H_
