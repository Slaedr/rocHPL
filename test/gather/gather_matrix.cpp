#include "gather/gather_matrix.hpp"

#include <cstdlib>
#include <limits>

#include <cnpy.h>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"

namespace ornl_hpl {
namespace test {

// gathers a distributed matrix to rank 0
std::vector<double> gather_matrix(const HPL_T_grid *const grid, const HPL_T_pmat *const mat)
{
    const int srcproc = 0;
    const int dest = 0;
    const int gl_size = mat->n * mat->n;
    double *gmat;
    if(grid->iam == dest) {
        hipMalloc(&gmat, gl_size*sizeof(double));
    }

    if(grid->iam == dest) {
        for(int src_proc_col = 0; src_proc_col < grid->npcol; src_proc_col++) {
          const int ncols = HPL_numroc(mat->n, mat->nb, mat->nb, src_proc_col,
                                               srcproc, grid->npcol);
          for(int src_proc_row = 0; src_proc_row < grid->nprow; src_proc_row++) {
            const int nrows = HPL_numroc(mat->n, mat->nb, mat->nb, src_proc_row,
                                                 srcproc, grid->nprow);
            double *rbuf;
            const int srcld = get_padded_dim(nrows);
            const int lsz = srcld * (ncols + 1);
            hipMalloc(&rbuf, lsz*sizeof(double));
            // mapping is column-major by default
            //const int src_rank = src_proc_row + src_proc_col*grid->nprow;
            const int src_rank = get_mpi_rank(grid, src_proc_row, src_proc_col);
            if(src_rank != dest) {
                MPI_Recv(rbuf, lsz, MPI_DOUBLE, src_rank, 0, grid->all_comm, MPI_STATUS_IGNORE);
            }

            // copy to global matrix
            for(int loc_col = 0; loc_col < ncols; loc_col += mat->nb) {
                const int gl_col = HPL_indxl2g(loc_col, mat->nb, mat->nb, src_proc_col,
                                               srcproc, grid->npcol);
                for(int loc_row = 0; loc_row < nrows; loc_row += mat->nb) {
                    const int gl_row = HPL_indxl2g(loc_row, mat->nb, mat->nb, src_proc_row,
                                                   srcproc, grid->nprow);
                    //gmat[loc_row + loc_col*mat->n] = rbuf[loc_row + loc_col * srcld];
                    if(src_rank != dest) {
                        hipMemcpy2D(gmat + gl_row + gl_col*mat->n, mat->n,
                            rbuf + loc_row + loc_col*srcld, srcld, mat->nb*sizeof(double), mat->nb,
                            hipMemcpyDeviceToDevice);
                    } else {
                        if(srcld != mat->ld) {
                            HPL_pabort(__LINE__, "gather_matrix", "Computer ld is not the same as mat->ld!");
                        }
                        hipMemcpy2D(gmat + gl_row + gl_col*mat->n, mat->n,
                            mat->dA + loc_row + loc_col*srcld, srcld, mat->nb*sizeof(double), mat->nb,
                            hipMemcpyDeviceToDevice);
                    }
                }
            }
          }
        }
    } else {
        MPI_Send(mat->dA, mat->ld*mat->nq, MPI_DOUBLE, dest, 0, grid->all_comm);
    }

    std::vector<double> host_gmat;
    if(grid->iam == dest) {
        host_gmat.resize(gl_size);
        hipMemcpy(host_gmat.data(), gmat, gl_size * sizeof(double), hipMemcpyDeviceToHost);
        hipFree(gmat);
    }

    return host_gmat;
}

template <typename scalar, typename file_scalar>
std::vector<scalar> read_block(const cnpy::NpyArray& arr)
{
    const file_scalar *const hA = arr.data<file_scalar>();
    const size_t buf_len = arr.shape[0]*arr.shape[1];
    std::vector<scalar> atemp(buf_len);

    // transpose if necesssary
    if(arr.fortran_order) {
        for(int j = 0; j < arr.shape[1]; j++) {
            for(int i = 0; i < arr.shape[0]; i++) {
                atemp[i + j*arr.shape[0]] = 
                    static_cast<scalar>(hA[i + j*arr.shape[0]]);
            }
        }
    } else {
        for(int j = 0; j < arr.shape[1]; j++) {
            for(int i = 0; i < arr.shape[0]; i++) {
                atemp[i + j*arr.shape[0]] =
                    static_cast<scalar>(hA[j + i*arr.shape[1]]);
            }
        }
    }
    return atemp;
}

std::vector<double> read_matrix_single_block(const std::string& path, const int global_size)
{
    std::vector<double> mat_vals;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        if(arr.shape[0] != global_size) {
            throw std::runtime_error("Mismatched num rows!");
        }
        if(arr.shape[1] != global_size) {
            throw std::runtime_error("Mismatched num cols!");
        }
        if(arr.word_size == 4) {
            std::cout << "Reading matrix with float values." << std::endl;
            auto vals = read_block<double,float>(arr);
            mat_vals = std::move(vals);
        } else if(arr.word_size == 8) {
            std::cout << "Reading matrix with double values." << std::endl;
            auto vals = read_block<double,double>(arr);
            mat_vals = std::move(vals);
        } else {
            ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(
                    "Matrix file has unsupported scalar type!");
        }
    }
    return mat_vals;
}

bool compare_gathered_matrices(const std::vector<double>& a, const std::vector<double>& b, const int num_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    if(rank == 0) {
        if(a.size() != num_rows*num_rows || b.size() != num_rows*num_rows) {
            throw std::runtime_error("Test failed: buffer sizes are unexpected.");
        }
        constexpr double eps = std::numeric_limits<double>::epsilon();
        for(int i = 0; i < num_rows * num_rows; i++) {
            if(std::abs(a[i]) < eps) {
                if(std::abs(a[i] - b[i]) >= eps) {
                    std::cout << "FAILED: Position " << i
                        << " has unequal entries which are very close to 0.\n";
                    result = 0;
                    break;
                }
            } else if(std::abs(a[i] - b[i])/std::abs(a[i]) >= 10*eps) {
                std::cout << "FAILED: Position " << i << " has unequal entries " << a[i] << " and "
                    << b[i] << ".\n";
                result = 0;
                break;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return static_cast<bool>(result);
}

}
}
