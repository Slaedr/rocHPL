#include <iostream>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"
#include "test_utils.hpp"

using namespace ornl_hpl;

void test_rank_flattening(const HPL_T_grid grid, const int flat_rank)
{
    if(grid.order == HPL_COLUMN_MAJOR) {
        const int local_size = grid.local_nprow * grid.local_npcol;
        const int tnode = flat_rank / local_size;
        const int tlocrank = flat_rank % local_size;

        const int local_mycol = tlocrank / grid.local_nprow;
        const int local_myrow = tlocrank % grid.local_nprow;

        const int noderow = tnode % (grid.nprow / grid.local_nprow);
        const int nodecol = tnode / (grid.nprow / grid.local_nprow);

        const int myrow = noderow * grid.local_nprow + local_myrow;
        const int mycol = nodecol * grid.local_npcol + local_mycol;

        HPL_ASSERT_THROW_SEQ(get_mpi_rank(&grid, myrow, mycol) == flat_rank);
    } else {
        ORNL_HPL_THROW_NOT_IMPLEMENTED("test row-major grid");
    }
}

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

    const int trank = get_mpi_rank(&grid, 0, 0);
    HPL_ASSERT_THROW_SEQ(trank == 0);

    // 1x8 local grid, 16 x 16 global grid
    {
        grid.nprow = grid.npcol = 16;
        grid.local_nprow = 1;
        grid.local_npcol = 8;
        grid.nprocs = 256;
        int trank = 66;
        test_rank_flattening(grid, trank);
        trank = 128;
        test_rank_flattening(grid, trank);
    }
    
    // 2x8 local grid, 8 x 32 global grid
    {
        grid.nprow = 8;
        grid.npcol = 32;
        grid.local_nprow = 2;
        grid.local_npcol = 8;
        grid.nprocs = 256;
        int trank = 53;
        test_rank_flattening(grid, trank);
        trank = 190;
        test_rank_flattening(grid, trank);
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}
