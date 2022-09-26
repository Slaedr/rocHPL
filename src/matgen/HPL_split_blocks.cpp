
#include "hpl.hpp"
#include "hpl_exceptions.hpp"

void split_blocks(HPL_T_test *const test, const HPL_T_palg *const algo,
                  const HPL_T_grid *const grid,
                  const HPL_T_pmat *const origmat, const int split_factor,
                  HPL_T_pmat *const mat)
{
    const int block_size = origmat->nb / split_factor;
    if(origmat->nb % split_factor != 0) {
        ORNL_HPL_THROW_NOT_SUPPORTED(
            "Requested block splitting factor must exactly divide initial block size.");
    }
    int ierr = HPL_pdmatgen(test, grid, algo, mat, origmat->n, block_size);
    if(ierr != HPL_SUCCESS) {
        throw std::runtime_error("split_blocks: New mat allocation failed!");
    }

    const int srcproc = 0;
    using scalar = double;
    const MPI_Datatype datatype = MPI_DOUBLE;
    //const int num_msgs_sent = origmat->nq / block_size * origmat->mp / blocksize;

    for(int loc_orig_col = 0; loc_orig_col < origmat->nq; loc_orig_col += block_size) {
        const int gl_col = HPL_indxl2g(loc_orig_col, origmat->nb, origmat->nb, grid->mycol,
                                       srcproc, grid->npcol);
        const int loc_new_col = HPL_indxg2l(gl_col, block_size, block_size, srcproc, grid->npcol);
        const int new_proc_col = HPL_indxg2p(gl_col, block_size, block_size, srcproc, grid->npcol);

        for(int loc_orig_row = 0; loc_orig_row < origmat->mp; loc_orig_row += block_size) {
            const int gl_row = HPL_indxl2g(loc_orig_row, origmat->nb, origmat->nb, grid->myrow,
                                           srcproc, grid->nprow);
            const int new_proc_row = HPL_indxg2p(gl_row, block_size, block_size, srcproc,
                                                 grid->nprow);
            const int loc_new_row = HPL_indxg2l(gl_row, block_size, block_size, srcproc,
                    grid->nprow);
            // Copy block into buffer
            scalar *buf{}, *d_buf{};
            hipHostMalloc(&buf, block_size*block_size*sizeof(scalar));
            hipMalloc(&d_buf, block_size*block_size*sizeof(scalar));
            device_copy_2d_block(origmat->ld, block_size, block_size,
                    origmat->dA + loc_orig_row + loc_orig_col * origmat->ld, block_size, d_buf);
            hipMemcpy(buf, d_buf, block_size*block_size*sizeof(scalar), hipMemcpyDeviceToHost);
            hipFree(d_buf);

            const int dest_rank = grid->order == HPL_ROW_MAJOR ?
                loc_new_row * grid->npcol + loc_new_col :
                loc_new_row + loc_new_col * grid->nprow;
            const int tag = gl_row + gl_col * origmat->n;
            // Send, and ye shall receive (in the next loop)
            MPI_Send(buf, block_size*block_size, datatype, dest_rank, tag, grid->all_comm);

            hipHostFree(buf);
        }
    }

    for(int loc_col = 0; loc_col < mat->nq; loc_col += block_size) {
        const int gl_col = HPL_indxl2g(loc_col, block_size, block_size, grid->mycol,
                                       srcproc, grid->npcol);
        const int loc_old_col = HPL_indxg2l(gl_col, origmat->nb, origmat->nb, srcproc, grid->npcol);
        const int old_proc_col = HPL_indxg2p(gl_col, origmat->nb, origmat->nb, srcproc, grid->npcol);

        for(int loc_row = 0; loc_row < mat->mp; loc_row += block_size) {
            const int gl_row = HPL_indxl2g(loc_row, block_size, block_size, grid->myrow,
                                           srcproc, grid->nprow);
            const int old_proc_row = HPL_indxg2p(gl_row, origmat->nb, origmat->nb, srcproc,
                                                 grid->nprow);
            const int loc_old_row = HPL_indxg2l(gl_row, origmat->nb, origmat->nb, srcproc,
                    grid->nprow);
            
            scalar *buf{}, *d_buf{};
            hipHostMalloc(&buf, block_size*block_size*sizeof(scalar));
            hipMalloc(&d_buf, block_size*block_size*sizeof(scalar));

            const int source_rank = grid->order == HPL_ROW_MAJOR ?
                loc_old_row * grid->npcol + loc_old_col :
                loc_old_row + loc_old_col * grid->nprow;
            const int tag = gl_row + gl_col * origmat->n;
            MPI_Recv(buf, block_size*block_size, datatype, source_rank, tag, grid->all_comm,
                    MPI_STATUS_IGNORE);
            
            hipMemcpy(buf, d_buf, block_size*block_size*sizeof(scalar), hipMemcpyDeviceToHost);
            hipHostFree(buf);
            device_copy_2d_block(origmat->ld, block_size, block_size,
                    origmat->dA + loc_row + loc_col * origmat->ld, block_size, d_buf);
            hipFree(d_buf);

        }
    }
}
