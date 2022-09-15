
#include <fstream>
#include <cnpy.h>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"

template <typename scalar>
void read_one_block(const int ibrow, const int jbcol, const cnpy::NpyArray& arr,
    const int block_size, const HPL_T_grid *const grid, HPL_T_pmat *const mat)
{
    const scalar *const hA = arr.data<scalar>();
    const int global_I = ibrow * block_size;
    const int global_J = jbcol * block_size;
    int local_i{-1}, local_j{-1};
    int owner_proc_i{-1}, owner_proc_j{-1};
    const int first_row_block_size = block_size;
    const int first_col_block_size = block_size;
    const int row_block_size = block_size;
    const int col_block_size = block_size;
    const int proc_row_for_I = grid->myrow;
    const int proc_col_for_J = grid->mycol;
    HPL_infog2l(global_I, global_J, first_row_block_size, row_block_size,
        first_col_block_size, col_block_size, proc_row_for_I, proc_col_for_J,
        grid->myrow, grid->mycol, grid->nprow, grid->npcol, &local_i, &local_j,
        &owner_proc_i, &owner_proc_j);
    const std::string err_msg =
        " inconsistent distribution while reading matrix on process ("
        + std::to_string(grid->myrow) + ", " + std::to_string(grid->mycol) + ")!";
    if(owner_proc_i != grid->myrow) {
        throw std::runtime_error("Row" + err_msg);
    }
    if(owner_proc_j != grid->mycol) {
        throw std::runtime_error("Column" + err_msg);
    }

    if(arr.fortran_order) {
        for(int j = local_j; j < block_size; j++) {
            for(int i = local_i; i < block_size; i++) {
                mat->A[i + j*mat->ld] = 
                    static_cast<double>(hA[i - local_i + (j - local_j)*arr.shape[0]]);
            }
        }
    } else {
        for(int j = local_j; j < block_size; j++) {
            for(int i = local_i; i < block_size; i++) {
                mat->A[i + j*mat->ld] =
                    static_cast<double>(hA[j - local_j + (i - local_i)*arr.shape[1]]);
            }
        }
    }
}

template <typename scalar>
void read_vector_redundant(const int ibrow, const cnpy::NpyArray& arr,
    const int block_size, const HPL_T_grid *const grid, HPL_T_pmat *const mat)
{
    const scalar *const hb = arr.data<scalar>();
    const int global_I = ibrow * block_size;
    const int global_J = mat->n;
    int local_i{-1}, local_j{-1};
    int owner_proc_i{-1}, owner_proc_j{-1};
    const int first_row_block_size = block_size;
    const int first_col_block_size = block_size;
    const int row_block_size = block_size;
    const int col_block_size = block_size;
    const int proc_row_for_I = grid->myrow;
    const int proc_col_for_J = grid->mycol;
    HPL_infog2l(global_I, global_J, first_row_block_size, row_block_size,
        first_col_block_size, col_block_size, proc_row_for_I, proc_col_for_J,
        grid->myrow, grid->mycol, grid->nprow, grid->npcol, &local_i, &local_j,
        &owner_proc_i, &owner_proc_j);
    if(local_j != mat->nq) {
        throw std::runtime_error("Inconsistent location of b vector storage!");
    }
    const std::string err_msg =
        " inconsistent distribution while reading b on process ("
        + std::to_string(grid->myrow) + ", " + std::to_string(grid->mycol) + ")!";
    if(owner_proc_i != grid->myrow) {
        throw std::runtime_error("Row" + err_msg);
    }
    if(owner_proc_j != grid->mycol) {
        throw std::runtime_error("Column" + err_msg);
    }

    if(arr.fortran_order) {
        for(int i = local_i; i < block_size; i++) {
            mat->A[i + local_j*mat->ld] = static_cast<double>(hb[i - local_i]);
        }
    } else {
        for(int i = local_i; i < block_size; i++) {
            mat->A[i + local_j*mat->ld] = static_cast<double>(hb[i - local_i]);
        }
    }
}

/*
 * The block size is fixed to that provided by the input.
 * The number of processors need not be the same as the number of blocks;
 * the number of processors must be at most equal to the number of blocks.
 */
void HPL_pdreadmat(const HPL_T_grid* const grid,
                   const int nrows_global,
                   const int ncols_global,
                   const int block_size,
                   const std::string path_prefix,
                   HPL_T_pmat* const mat)
{
    const std::string desc_file = path_prefix + "/desc.txt";
    std::string header;
    std::ifstream desc_stream(desc_file);
    std::getline(desc_stream, header);
    int n_block_rows{-1}, n_block_cols{-1}, n_total_rows{}, n_rhs{};
    desc_stream >> n_block_rows >> n_block_cols >> n_total_rows >> n_rhs;
    if(grid->iam == 0) {
        std::cout << "Matrix contains " << n_block_rows << ", " << n_block_cols
            << " blocks; size = " << n_total_rows << ", num RHS = " << n_rhs
            << std::endl;
    }
    desc_stream.close();
    if(n_block_rows < grid->nprow) {
        throw std::runtime_error("Not enough block-rows!");
    }
    if(n_block_cols < grid->npcol) {
        throw std::runtime_error("Not enough block-columns!");
    }

    for(int ibrow = grid->myrow; ibrow < n_block_rows; ibrow += grid->nprow) {
        for(int jbcol = grid->mycol; jbcol < n_block_cols; jbcol += grid->npcol) {
            const std::string path = path_prefix + "/A_" + std::to_string(ibrow) + "_"
                + std::to_string(jbcol) + ".npy";
            cnpy::NpyArray arr = cnpy::npy_load(path);
            if(arr.word_size == 4) {
                read_one_block<float>(ibrow, jbcol, arr, block_size, grid, mat);
            } else if(arr.word_size == 8) {
                read_one_block<double>(ibrow, jbcol, arr, block_size, grid, mat);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "Matrix file has unsupported scalar type!");
            }
        }
    }

    // b vector stored in the last process-column
    // NOTE: For now, we assume b is stored in a single file and redundantly load it on all
    //  processes in the last process-column.
    if(grid->mycol == grid->npcol - 1) {
        const std::string path = path_prefix + "/b.npy";
        cnpy::NpyArray arr = cnpy::npy_load(path);
        for(int ibrow = grid->myrow; ibrow < n_block_rows; ibrow += grid->nprow) {
            //
            if(arr.word_size == 4) {
                read_vector_redundant<float>(ibrow, arr, block_size, grid, mat);
            } else if(arr.word_size == 8) {
                read_vector_redundant<double>(ibrow, arr, block_size, grid, mat);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "b-vector file has unsupported scalar type!");
            }
        }
    }
}

