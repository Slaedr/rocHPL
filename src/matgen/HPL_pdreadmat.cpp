
#include <fstream>
#include <iostream>
#include <cnpy.h>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"


template <typename scalar, typename file_scalar>
void read_one_block(const int ibrow, const int jbcol, const cnpy::NpyArray& arr,
    const int block_size, const HPL_T_grid *const grid, const int mat_ld, scalar *const mat)
{
    const file_scalar *const hA = arr.data<file_scalar>();
    const int global_I = ibrow * block_size;
    const int global_J = jbcol * block_size;
    int local_i{-1}, local_j{-1};

    int owner_proc_i{-1}, owner_proc_j{-1};
    const int first_row_block_size = block_size;
    const int first_col_block_size = block_size;
    const int row_block_size = block_size;
    const int col_block_size = block_size;
    const int proc_row_start = 0;
    const int proc_col_start = 0;
    HPL_infog2l(global_I, global_J, first_row_block_size, row_block_size,
        first_col_block_size, col_block_size, proc_row_start, proc_col_start,
        grid->myrow, grid->mycol, grid->nprow, grid->npcol, &local_i, &local_j,
        &owner_proc_i, &owner_proc_j);

    {
        // test
        const std::string err_msg =
            " inconsistent distribution while reading mat on process ("
            + std::to_string(grid->myrow) + ", " + std::to_string(grid->mycol) + ")!";

        const bool row_is_present = (ibrow % grid->nprow == grid->myrow);
        if(!row_is_present) {
            std::cout << "A: Proc " << grid->myrow << ", " << grid->mycol << ": "
                << local_i << " " << local_j << " " << owner_proc_i << " " << owner_proc_j
                << std::endl;
            throw std::runtime_error("Row not present!");
        }
        if(owner_proc_i != grid->myrow) {
            throw std::runtime_error("Row not present 2!");
        }
        const int t_bloc_i = ibrow / grid->nprow;
        if(t_bloc_i*block_size != local_i) {
            throw std::runtime_error("Row " + err_msg);
        }

        const bool col_is_present = (jbcol % grid->npcol == grid->mycol);
        if(!col_is_present) {
            throw std::runtime_error("Col not present!");
        }
        if(owner_proc_j != grid->mycol) {
            throw std::runtime_error("Col not present 2!");
        }
        const int t_bloc_j = jbcol / grid->npcol;
        if(t_bloc_j*block_size != local_j) {
            throw std::runtime_error("Col " + err_msg);
        }
    }

    const size_t bufsize = arr.shape[0]*arr.shape[1]*sizeof(scalar);
    scalar *atemp{};
    hipHostMalloc(&atemp, bufsize);

    // transpose if necesssary
    if(arr.fortran_order) {
        for(int j = 0; j < block_size; j++) {
            for(int i = 0; i < block_size; i++) {
                atemp[i + j*arr.shape[0]] = 
                    static_cast<scalar>(hA[i + j*arr.shape[0]]);
            }
        }
    } else {
        for(int j = 0; j < block_size; j++) {
            for(int i = 0; i < block_size; i++) {
                atemp[i + j*arr.shape[0]] =
                    static_cast<scalar>(hA[j + i*arr.shape[1]]);
            }
        }
    }

    scalar *d_atemp;
    hipMalloc(&d_atemp, bufsize);
    hipMemcpy(d_atemp, atemp, bufsize, hipMemcpyHostToDevice);

    device_copy_2d_block(arr.shape[0], arr.shape[0], arr.shape[1], d_atemp,
        mat_ld, mat + local_i + local_j*mat_ld);

    hipHostFree(atemp);
    hipFree(d_atemp);
}

template <typename scalar, typename file_scalar>
void read_vector_redundant(const int ibrow, const cnpy::NpyArray& arr,
    const int block_size, const HPL_T_grid *const grid, HPL_T_pmat *const mat)
{
    const file_scalar *const hb = arr.data<file_scalar>();
    const int global_I = ibrow * block_size;
    const int global_J = mat->n;
    int local_i{-1}, local_j{-1};
    int owner_proc_i{-1}, owner_proc_j{-1};
    const int first_row_block_size = block_size;
    const int first_col_block_size = block_size;
    const int row_block_size = block_size;
    const int col_block_size = block_size;
    const int proc_row_start = 0;
    const int proc_col_start = 0;
    HPL_infog2l(global_I, global_J, first_row_block_size, row_block_size,
        first_col_block_size, col_block_size, proc_row_start, proc_col_start,
        grid->myrow, grid->mycol, grid->nprow, grid->npcol, &local_i, &local_j,
        &owner_proc_i, &owner_proc_j);
    if(local_j != mat->nq-1) {
        std::cout << "b: Proc " << grid->myrow << ", " << grid->mycol << ": "
            << local_i << " " << local_j << " " << owner_proc_i << " " << owner_proc_j
            << ", local nq = " << mat->nq << std::endl;
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
    
    const size_t bufsize = block_size*sizeof(scalar);
    // need intermediate buffer for casting
    scalar *btemp{};
    hipHostMalloc(&btemp, bufsize);

    for(int i = 0; i < block_size; i++) {
        btemp[i] = static_cast<scalar>(hb[ibrow*block_size + i]);
    }
    hipMemcpy(mat->dA + local_i + local_j * mat->ld, btemp, bufsize, hipMemcpyHostToDevice);
    hipHostFree(btemp);
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
    const int n_total_cols = n_total_rows;

    //if(n_block_rows != grid->nprow) {
    //    ORNL_HPL_THROW_NOT_IMPLEMENTED("Grid nprow should match number of block-rows in files.");
    //}
    //if(n_block_cols != grid->npcol) {
    //    ORNL_HPL_THROW_NOT_IMPLEMENTED("Grid npcol should match number of block-cols in files.");
    //}

    const int file_block_row_size = n_total_rows / n_block_rows;
    const int file_block_col_size = n_total_cols / n_block_cols;
    if(file_block_row_size != file_block_col_size) {
        ORNL_HPL_THROW_NOT_SUPPORTED("Rectangular blocks.");
    }
    if(file_block_row_size != block_size) {
        ORNL_HPL_THROW_NOT_IMPLEMENTED(
            "Different block size requested from the one in input files.");
    }

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
                read_one_block<double,float>(ibrow, jbcol, arr, block_size, grid,
                                             mat->ld, mat->dA);
            } else if(arr.word_size == 8) {
                read_one_block<double,double>(ibrow, jbcol, arr, block_size, grid,
                                              mat->ld, mat->dA);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "Matrix file has unsupported scalar type!");
            }
        }
    }

    // b vector
    // NOTE: For now, we assume b is stored in a single file and redundantly load it on all
    //  processes in the proc-column the contains the last matrix column
    if(grid->mycol == HPL_indxg2p(n_total_cols, block_size, block_size, 0, grid->npcol)) {
        // If this proc-column contains the (N+1)th column, read b
        const std::string b_path = path_prefix + "/b.npy";
        cnpy::NpyArray arr = cnpy::npy_load(b_path);
        for(int ibrow = grid->myrow; ibrow < n_block_rows; ibrow += grid->nprow) {
            //
            if(arr.word_size == 4) {
                read_vector_redundant<double,float>(ibrow, arr, block_size, grid, mat);
            } else if(arr.word_size == 8) {
                read_vector_redundant<double,double>(ibrow, arr, block_size, grid, mat);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "b-vector file has unsupported scalar type!");
            }
        }
    }
}

void HPL_gather_solution(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
    const std::string& matrix_dir)
{
    if(grid->myrow == 0) {
        // gather X
        double *hX{};
        if(grid->mycol == 0) {
            hX = static_cast<double*>(malloc(mat->n * sizeof(double)));
        }
        auto hX_piece = static_cast<double*>(malloc(mat->nq * sizeof(double)));
        hipMemcpy(hX_piece, mat->dX, mat->nq * sizeof(double), hipMemcpyDeviceToHost);
        // TODO: block-cyclic gather into hX
        //
        free(hX_piece);
        if(grid->mycol == 0) {
            cnpy::npy_save(matrix_dir + "/x_sol.npy", hX,
                std::vector<size_t>{static_cast<size_t>(mat->n), 0});
            free(hX);
        }
    }
}

