
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

#include <cnpy.h>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"


template <typename scalar, typename file_scalar>
void read_one_block(const int ibrow, const int jbcol, const cnpy::NpyArray& arr,
    const int block_size, const HPL_T_grid *const grid, const int mat_ld, scalar *const mat)
{
    const file_scalar *const hA = arr.data<file_scalar>();
    if(block_size != arr.shape[0]) {
        throw std::runtime_error("block size requested should be same as file!");
    }
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
        const int test_owner_proc_i = HPL_indxg2p(global_I, block_size, block_size, 0, grid->nprow);
        const int test_owner_proc_j = HPL_indxg2p(global_J, block_size, block_size, 0, grid->npcol);
        if(test_owner_proc_i != owner_proc_i) {
            throw std::runtime_error("Inconsistent_rows");
        }
        if(test_owner_proc_j != owner_proc_j) {
            throw std::runtime_error("Inconsistent_cols");
        }
        const int test_owner_proc_e_i = HPL_indxg2p(global_I+block_size-1, block_size, block_size,
                0, grid->nprow);
        const int test_owner_proc_e_j = HPL_indxg2p(global_J+block_size-1, block_size, block_size,
                0, grid->npcol);
        if(test_owner_proc_e_i != owner_proc_i) {
            throw std::runtime_error("Inconsistent proc rows for end of block");
        }
        if(test_owner_proc_e_j != owner_proc_j) {
            throw std::runtime_error("Inconsistent proc cols for end of block");
        }
        const int test_local_i = HPL_indxg2l(global_I, block_size, block_size, 0, grid->nprow);
        const int test_local_j = HPL_indxg2l(global_J, block_size, block_size, 0, grid->npcol);
        if(test_local_i != local_i) {
            throw std::runtime_error("Inconsistent_local rows");
        }
        if(test_local_j != local_j) {
            throw std::runtime_error("Inconsistent_local cols");
        }
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
    //hipHostMalloc(&atemp, bufsize);
    atemp = static_cast<scalar*>(malloc(bufsize));

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

    hipMemcpy2D(mat + local_i + local_j*mat_ld, mat_ld*sizeof(scalar),
            atemp, arr.shape[0]*sizeof(scalar), block_size*sizeof(scalar), block_size,
            hipMemcpyHostToDevice);

    //hipHostFree(atemp);
    free(atemp);
}

template <typename scalar, typename file_scalar>
void read_vector_redundant(const int ibrow, const cnpy::NpyArray& arr,
    const HPL_T_grid *const grid, HPL_T_pmat *const mat)
{
    const int block_size = mat->nb;
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

    {
        //test
        const int test_proc_i = HPL_indxg2p(global_I, mat->nb, mat->nb, 0, grid->nprow);
        if(test_proc_i != owner_proc_i) {
            throw std::runtime_error("Inconsistent owner row proc while reading b!");
        }
        const int test_local_i = HPL_indxg2l(global_I, mat->nb, mat->nb, 0, grid->nprow);
        if(test_local_i != local_i) {
            throw std::runtime_error("Inconsistent local row idx while reading b!");
        }
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
    }
    
    const size_t bufsize = block_size*sizeof(scalar);
    // need intermediate buffer for casting
    auto btemp = static_cast<scalar*>(malloc(bufsize));
    //hipHostMalloc(&btemp, bufsize);

    for(int i = 0; i < block_size; i++) {
        btemp[i] = static_cast<scalar>(hb[ibrow*block_size + i]);
    }
    hipMemcpy(mat->dA + local_i + local_j * mat->ld, btemp, bufsize, hipMemcpyHostToDevice);
    //hipHostFree(btemp);
    free(btemp);
}

namespace {

std::string get_matrix_file_name(int ibrow, int jbcol, ornl_hpl::matrix_dir_type mdtype);

}

/*
 * The block size is fixed to that provided by the input.
 * The number of processors need not be the same as the number of blocks;
 * the number of processors must be at most equal to the number of blocks.
 */
void HPL_pdreadmat(const HPL_T_grid* const grid,
                   const int nrows_global,
                   const int ncols_global,
                   const std::string path_prefix,
                   const ornl_hpl::matrix_dir_type mdtype,
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
    //MPI_Barrier(grid->all_comm);
    const int n_total_cols = n_total_rows;
    const int block_size = mat->nb;

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
        std::cout << "File row block size = " << file_block_row_size <<
            ", block size = " << block_size << std::endl;
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
            const std::string path = path_prefix + get_matrix_file_name(ibrow, jbcol, mdtype);
            cnpy::NpyArray arr = cnpy::npy_load(path);
            if(arr.word_size == 4) {
                if(ibrow == 0 && jbcol == 0) {
                    std::cout << "Reading matrix with float values." << std::endl;
                }
                read_one_block<double,float>(ibrow, jbcol, arr, block_size, grid,
                                             mat->ld, mat->dA);
            } else if(arr.word_size == 8) {
                if(ibrow == 0 && jbcol == 0) {
                    std::cout << "Reading matrix with double values." << std::endl;
                }
                read_one_block<double,double>(ibrow, jbcol, arr, block_size, grid,
                                              mat->ld, mat->dA);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "Matrix file has unsupported scalar type!");
            }
        }
    }

    MPI_Barrier(grid->all_comm);
    if(grid->iam == 0) {
        printf("Read all matrix blocks.\n"); fflush(stdout);
    }

    // b vector
    // NOTE: For now, we assume b is stored in a single file and redundantly load it on all
    //  processes in the proc-column that contains the last matrix column
    if(grid->mycol == HPL_indxg2p(n_total_cols, block_size, block_size, 0, grid->npcol)) {
        // If this proc-column contains the (N+1)th column, read b
        const std::string b_path = path_prefix + "/b.npy";
        cnpy::NpyArray arr = cnpy::npy_load(b_path);
        {
            if(arr.shape[0] != block_size * n_block_rows) {
                std::string err = " Reading b vector: " + std::to_string(arr.shape[0])
                    + " vs " + std::to_string(block_size * n_block_rows);
                ORNL_HPL_THROW_INCONSISTENT_BLOCK_SIZE(err);
            }
            if(arr.shape.size() != 1 && arr.shape[1] != 1) {
                throw std::runtime_error("Multiple RHS not supported!");
            }
        }
        for(int ibrow = grid->myrow; ibrow < n_block_rows; ibrow += grid->nprow) {
            //
            if(arr.word_size == 4) {
                read_vector_redundant<double,float>(ibrow, arr, grid, mat);
            } else if(arr.word_size == 8) {
                read_vector_redundant<double,double>(ibrow, arr, grid, mat);
            } else {
                ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                        "b-vector file has unsupported scalar type!");
            }
        }
    }

    MPI_Barrier(grid->all_comm);
    if(grid->iam == 0) {
        printf("Read all vector blocks.\n"); fflush(stdout);
    }
}

namespace {

template <typename scalar>
void copy_blocks(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
    const scalar *const local_x, const int remote_proc, const int remote_size,
    scalar *const global_x);

}

void HPL_gather_solution(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
                         double *const hX)
{
    using scalar = double;
    const int root = 0;
    const int local_nq = mat->nq-1; // account for extra column
    
    scalar *hlocX{};
    //hipHostMalloc(&hlocX, local_nq * sizeof(scalar));
    hlocX = static_cast<scalar*>(malloc(local_nq * sizeof(scalar)));
    ORNL_HPL_CHECK_ALLOC(hlocX, "host");
    hipMemcpy(hlocX, mat->dX, local_nq * sizeof(scalar), hipMemcpyDeviceToHost);
    if(grid->mycol != 0) {
        MPI_Send(hlocX, local_nq, MPI_DOUBLE, root, 1, grid->row_comm);
    }

    {
        const int lastpcol = HPL_indxg2p(mat->n-1, mat->nb, mat->nb, 0, grid->npcol);
        if(grid->mycol == lastpcol) {
            const int locidx = HPL_indxg2l(mat->n-1, mat->nb, mat->nb, 0, grid->npcol);
            std::cout << "Last few entries of solution vector:"
                << std::endl;
            for(int i = std::max(0,locidx - 5); i <= locidx; i++) {
                std::cout << hlocX[i] << std::endl;
            }
            std::cout << std::endl;
        }
    }

    if(grid->mycol == root) {
        for(int jqcol = 0; jqcol < grid->npcol; jqcol++) {
            // For remote rank, compute nq
            // Sure we don't need to subtract 1 below?
            const int remote_nq = HPL_numroc(mat->n, mat->nb, mat->nb, jqcol, 0, grid->npcol);
            if(remote_nq % mat->nb != 0) {
                throw std::runtime_error("Invalid blocking during solution gather!");
            }

            scalar *hremX{};
            if(jqcol != root) {
                //hipHostMalloc(&hremX, remote_nq * sizeof(scalar));
                hremX = static_cast<scalar*>(malloc(remote_nq * sizeof(scalar)));
                ORNL_HPL_CHECK_ALLOC(hremX, "host");
                MPI_Recv(hremX, remote_nq, MPI_DOUBLE, jqcol, 1, grid->row_comm, MPI_STATUS_IGNORE);
            } else {
                hremX = hlocX;
                if(remote_nq != local_nq) {
                    std::cout << "Remote nq = " << remote_nq << ", this nq = " << local_nq
                        << std::endl;
                    throw std::runtime_error("Iconsistent local sizes in solution gather!");
                }

                std::cout << "First few entries of solution vector:" << std::endl;
                for(int i = 0; i < std::min(5, local_nq); i++) {
                    std::cout << hlocX[i] << std::endl;
                }
                std::cout << std::endl;
            }

            // Compute global indices knowing block size and remote proc-column index.
            copy_blocks(grid, mat, hremX, jqcol, remote_nq, hX);

            if(jqcol != root) {
                //hipHostFree(hremX);
                free(hremX);
            }
        }
    }

    //hipHostFree(hlocX);
    free(hlocX);
}

void HPL_gather_write_solution(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
                               const std::string& matrix_dir)
{
    using scalar = double;

    if(grid->myrow != 0) {
        return;
    }
        
    double *hX{};
    if(grid->mycol == 0) {
        hX = static_cast<scalar*>(malloc(mat->n * sizeof(scalar)));
    }

    HPL_gather_solution(grid, mat, hX);

    if(grid->mycol == 0) {
        //std::cout << "First few entries of gathered solution vector:" << std::endl;
        //for(int i = 0; i < std::min(5, mat->n); i++) {
        //    std::cout << hX[i] << std::endl;
        //}
        //std::cout << "Middle few entries of gathered solution vector:" << std::endl;
        //for(int i = mat->n/2; i < std::min(mat->n/2 + 5, mat->n); i++) {
        //    std::cout << hX[i] << std::endl;
        //}
        //std::cout << "Last few entries of gathered solution vector:" << std::endl;
        //for(int i = std::max(0, mat->n - 10); i < mat->n; i++) {
        //    std::cout << hX[i] << std::endl;
        //}
        //std::cout << std::endl;
        cnpy::npy_save(matrix_dir + "/solution/x.npy", hX,
            std::vector<size_t>{static_cast<size_t>(mat->n), 1});
        free(hX);
    }
}

void HPL_write_solution_by_blocks(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
                               const std::string& matrix_dir)
{
    using scalar = double;
    const int srcproc = 0;
    const int bs = mat->nb;
    const int nloccols = mat->nq-1; // account for extra column

    if(grid->myrow != 0) {
        return;
    }

    // copy local part to host
    auto hX = static_cast<scalar*>(malloc(nloccols * sizeof(scalar)));
    hipMemcpy(hX, mat->dX, nloccols * sizeof(scalar), hipMemcpyDeviceToHost);
       
    // iterate over local blocks
    for(int jloc = 0; jloc < nloccols; jloc += bs) {
        const int gl_col = HPL_indxl2g(jloc, bs, bs, grid->mycol, srcproc, grid->npcol);
        const int gl_blockcol = gl_col / bs;
        cnpy::npy_save(matrix_dir + "/solution/hpl/x_" + std::to_string(gl_blockcol) + ".npy",
            hX + jloc, std::vector<size_t>{static_cast<size_t>(bs), 1});
    }

    free(hX);
}

namespace {

std::string get_matrix_file_name(int ibrow, int jbcol, ornl_hpl::matrix_dir_type mdtype)
{
    if(mdtype == ornl_hpl::matrix_dir_type::row_block_dirs) {
        //std::stringstream str;
        //str << "row_" << std::setw(5) << std::setfill('0') << ibrow
        //    << "/A_" << jbcol << ".npy";
        //std::string path = str.str();
        std::string path = "/row_" + std::to_string(ibrow) + "/A_" + std::to_string(ibrow)
            + "_" + std::to_string(jbcol) + ".npy";
        return path;
    } else if(mdtype == ornl_hpl::matrix_dir_type::flat) {
        std::string path = "/A_" + std::to_string(ibrow) + "_" + std::to_string(jbcol)
            + ".npy";
        return path;
    } else {
        ORNL_HPL_THROW_NOT_SUPPORTED("Non-existent matrix directory layout.");
    }
}

template <typename scalar>
void copy_blocks(const HPL_T_grid *const grid, const HPL_T_pmat *const mat,
                 const scalar *const local_x, const int remote_proc, const int remote_size,
                 scalar *const global_x)
{
    constexpr int srcproc = 0;
    for(int lc = 0; lc < remote_size; lc += mat->nb) {
        const int gc = HPL_indxl2g(lc, mat->nb, mat->nb, remote_proc, srcproc, grid->npcol);
        std::copy(local_x + lc, local_x + lc + mat->nb, global_x + gc);
    }
}

}

