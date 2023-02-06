#include "gather/gather_matrix.hpp"

#include <cstdlib>
#include <string>
#include <iostream>

#include "hpl.hpp"
#include "test_utils.hpp"

using namespace ornl_hpl;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    HPL_T_test test;
    HPL_T_grid grid;

    // Prepare grid
    test::TestData td = test::initialize_grid_and_test(argc, argv, &grid, &test);
    
    const auto single_mat = test::read_matrix_single_block(td.reference_solution_path, td.global_size);
 
    HPL_T_pmat mat; 
    int ierr = HPL_pdmatgen(&test, &grid, &mat, td.global_size, td.block_size);
    if(ierr != HPL_SUCCESS) {
        HPL_pdmatfree(&mat);
        throw std::bad_alloc();
    }
 
    HPL_pdreadmat(&grid, td.global_size, td.global_size + 1, test.matrix_dir, test.mdtype, &mat);

    const auto gl_mat_vals = test::gather_matrix(&grid, &mat);

    if(rank == 0) {
        std::cout << "Sizes of: gathered mat = " << gl_mat_vals.size() << ", ref mat = "
            << single_mat.size() << ", global size = " << td.global_size << std::endl;
    }
    if(test::compare_gathered_matrices(gl_mat_vals, single_mat, td.global_size)) {
        MPI_Finalize();
        return 0;
    } else {
        MPI_Finalize();
        return -1;
    }

    MPI_Finalize();
    return -2;
}

