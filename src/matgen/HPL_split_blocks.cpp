#include <iostream>
#include <vector>

#include "hpl.hpp"
#include "hpl_exceptions.hpp"

/*
 * We assume uniform blocks and integer divisibility.
 */
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

    const int srcproc = 0;
    const int gl_size = origmat->n;
    using scalar = double;
    const MPI_Datatype datatype = MPI_DOUBLE;

    if(origmat->mp != mat->mp) {
        printf("WARNING: Original and new matrices have different mp on rank %d!", grid->iam);
    }
    if(origmat->nq != mat->nq) {
        printf("WARNING: Original and new matrices have different nq on rank %d!", grid->iam);
    }
    printf("Rank %d starting spltting..\n", grid->iam);
    fflush(stdout);

    for(int loc_col = 0; loc_col < origmat->nq-1; loc_col += block_size) {
        // originally, this local location has a certain global idx
        const int gl_col = HPL_indxl2g(loc_col, origmat->nb, origmat->nb, grid->mycol,
                                       srcproc, grid->npcol);
        // That global col idx maps to some local col in some other proc
        const int loc_new_col = HPL_indxg2l(gl_col, block_size, block_size, srcproc, grid->npcol);
        const int new_proc_col = HPL_indxg2p(gl_col, block_size, block_size, srcproc, grid->npcol);
       
        // In the split matrix, this local location maps to some (other) global col idx 
        const int new_gl_col = HPL_indxl2g(loc_col, block_size, block_size, grid->mycol,
                                       srcproc, grid->npcol);
        // That global col idx maps to some local idx in the original matrix on some other proc
        const int loc_old_col = HPL_indxg2l(new_gl_col, origmat->nb, origmat->nb, srcproc, grid->npcol);
        const int old_proc_col = HPL_indxg2p(new_gl_col, origmat->nb, origmat->nb, srcproc, grid->npcol);

        for(int loc_row = 0; loc_row < origmat->mp; loc_row += block_size) {
            // info needed for sending
            const int gl_row = HPL_indxl2g(loc_row, origmat->nb, origmat->nb, grid->myrow,
                                           srcproc, grid->nprow);
            const int new_proc_row = HPL_indxg2p(gl_row, block_size, block_size, srcproc,
                                                 grid->nprow);
            const int loc_new_row = HPL_indxg2l(gl_row, block_size, block_size, srcproc,
                    grid->nprow);
           
            // info needed for receiving 
            const int new_gl_row = HPL_indxl2g(loc_row, block_size, block_size, grid->myrow,
                                           srcproc, grid->nprow);
            const int old_proc_row = HPL_indxg2p(new_gl_row, origmat->nb, origmat->nb, srcproc,
                                                 grid->nprow);
            const int loc_old_row = HPL_indxg2l(new_gl_row, origmat->nb, origmat->nb, srcproc,
                    grid->nprow);
            
            const int dest_rank = grid->order == HPL_ROW_MAJOR ?
                new_proc_row * grid->npcol + new_proc_col :
                new_proc_row + new_proc_col * grid->nprow;
            const int loc_new_nrows = HPL_numroc(gl_size, block_size, block_size, new_proc_row,
                                                 srcproc, grid->nprow);
            const int tag = loc_new_row/block_size +
                loc_new_col/block_size * loc_new_nrows/block_size;
                
            scalar *d_buf{};
            hipMalloc(&d_buf, block_size*block_size*sizeof(scalar));
            MPI_Request send_req;

            if(dest_rank != grid->iam) {
                // Copy block into buffer
#ifdef HPL_MPI_NOT_GPU_AWARE
                scalar *buf{};
                buf = static_cast<scalar*>(malloc(block_size*block_size*sizeof(scalar)));
#endif
                //hipMemcpy2D(d_buf + buf_row + buf_col*buf_ld, buf_ld * sizeof(scalar),
                //    origmat->dA + loc_row + loc_col*origmat->ld,
                //    origmat->ld*sizeof(scalar), block_size * sizeof(scalar), block_size,
                //    hipMemcpyDeviceToDevice);
                HPL_device_copy_2d_to_array(d_buf, origmat->ld, block_size, block_size,
                    origmat->dA + loc_row + loc_col*origmat->ld);
                //if(buf_col >= origmat->nq) {
                //    ORNL_HPL_THROW_INSUFFICIENT_ALLOC("gpu");
                //}

                // Send, and ye shall receive (in the next loop)
                MPI_Isend(d_buf, block_size*block_size, datatype, dest_rank, tag, grid->all_comm, &send_req);
                printf("Rank %d returned from ISend.\n", grid->iam); fflush(stdout);

#ifdef HPL_MPI_NOT_GPU_AWARE
                free(buf);
#endif
            }

            const int source_rank = grid->order == HPL_ROW_MAJOR ?
                old_proc_row * grid->npcol + old_proc_col :
                old_proc_row + old_proc_col * grid->nprow;
            const int rec_tag = loc_row/block_size + loc_col/block_size * mat->mp / block_size;
            if(source_rank != grid->iam) {
                //
                scalar *d_rbuf{};
                hipMalloc(&d_rbuf, block_size*block_size*sizeof(scalar));
                MPI_Request req;
                MPI_Irecv(d_rbuf, block_size*block_size, datatype, source_rank, tag, grid->all_comm,
                          &req);
                printf("Rank %d returned from Irecv.\n", grid->iam); fflush(stdout);
                int ierr = MPI_Wait(&req, MPI_STATUS_IGNORE);
                if(ierr != MPI_SUCCESS) {
                    printf("IRecv failed on rank %d!\n", grid->iam); fflush(stdout);
                }
                printf("Rank %d completed Irecv.\n", grid->iam); fflush(stdout);
                
#ifdef HPL_MPI_NOT_GPU_AWARE
                free(buf);
#endif
                hipMemcpy2D(mat->dA + loc_row + loc_col*mat->ld, mat->ld*sizeof(scalar), d_rbuf,
                    block_size*sizeof(scalar), block_size*sizeof(scalar), block_size,
                    hipMemcpyDeviceToDevice);
                hipFree(d_rbuf);
            } else {
                hipMemcpy2D(mat->dA + loc_row + loc_col*mat->ld, mat->ld*sizeof(scalar),
                    origmat->dA + loc_old_row + loc_old_col*origmat->ld, origmat->ld*sizeof(scalar),
                    block_size * sizeof(scalar), block_size, hipMemcpyDeviceToDevice);
            }

            if(dest_rank != grid->iam) {
                int ierr = MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                if(ierr != MPI_SUCCESS) {
                    printf("ISend failed on rank %d!\n", grid->iam); fflush(stdout);
                }
                printf("Rank %d completed Isend.\n", grid->iam); fflush(stdout);
            }
            hipFree(d_buf);
        }
    }

#if 0
    for(int loc_col = 0; loc_col < mat->nq-1; loc_col += block_size) {
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

            const int source_rank = grid->order == HPL_ROW_MAJOR ?
                old_proc_row * grid->npcol + old_proc_col :
                old_proc_row + old_proc_col * grid->nprow;
            const int tag = loc_row/block_size + loc_col/block_size * mat->mp / block_size;
           
            if(source_rank != grid->iam) { 
#ifdef HPL_MPI_NOT_GPU_AWARE
                scalar *buf = static_cast<scalar*>(malloc(block_size*block_size*sizeof(scalar)));
#endif
                scalar *d_buf{};
                hipMalloc(&d_buf, block_size*block_size*sizeof(scalar));
                MPI_Request req;
                MPI_Irecv(d_buf, block_size*block_size, datatype, source_rank, tag, grid->all_comm,
                          &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
                
#ifdef HPL_MPI_NOT_GPU_AWARE
                free(buf);
#endif
                hipMemcpy2D(mat->dA + loc_row + loc_col*mat->ld, mat->ld*sizeof(scalar), d_buf,
                    block_size*sizeof(scalar), block_size*sizeof(scalar), block_size,
                    hipMemcpyDeviceToDevice);
                hipFree(d_buf);
            } else {
                hipMemcpy2D(mat->dA + loc_row + loc_col*mat->ld, mat->ld*sizeof(scalar),
                    origmat->dA + loc_old_row + loc_old_col*origmat->ld, origmat->ld*sizeof(scalar),
                    block_size * sizeof(scalar), block_size, hipMemcpyDeviceToDevice);
            }

        }
    }
#endif

    if(grid->myrow == 0 && grid->mycol == 0) {
        std::cout << "Completed splitting matrix. Splitting b vector..\n" << std::flush;
    }

    // split RHS vector
    const int b_gl_col = gl_size;
    const int orig_b_proc_col = HPL_indxg2p(b_gl_col, origmat->nb, origmat->nb, srcproc,
                                            grid->npcol);
    const int new_b_proc_col = HPL_indxg2p(b_gl_col, block_size, block_size, srcproc, grid->npcol);
    const int loc_b_orig_col = HPL_indxg2l(b_gl_col, origmat->nb, origmat->nb, srcproc, grid->npcol);
    const int loc_b_new_col = HPL_indxg2l(b_gl_col, block_size, block_size, srcproc, grid->npcol);
   
    if(grid->mycol == orig_b_proc_col) {
        if(loc_b_orig_col != origmat->nq-1) {
            throw std::runtime_error("Inconsistent b location in orig matrix");
        }

        for(int loc_orig_row = 0; loc_orig_row < origmat->mp; loc_orig_row += block_size) {
            const int gl_row = HPL_indxl2g(loc_orig_row, origmat->nb, origmat->nb, grid->myrow,
                                           srcproc, grid->nprow);
            const int new_proc_row = HPL_indxg2p(gl_row, block_size, block_size, srcproc,
                                                 grid->nprow);
            const int loc_new_row = HPL_indxg2l(gl_row, block_size, block_size, srcproc,
                                                grid->nprow);
            // Copy block into buffer
            //scalar *buf{};
            //buf = static_cast<scalar*>(malloc(block_size*sizeof(scalar)));
            //hipHostMalloc(&buf, block_size*sizeof(scalar));
            //hipMemcpy(buf, origmat->dA + loc_orig_row + origmat->ld * loc_b_orig_col,
            //          block_size*sizeof(scalar), hipMemcpyDeviceToHost);

            const int dest_rank = grid->order == HPL_ROW_MAJOR ?
                new_proc_row * grid->npcol + new_b_proc_col :
                new_proc_row + new_b_proc_col * grid->nprow;
            const int loc_new_nrows = HPL_numroc(gl_size, block_size, block_size, new_proc_row,
                                                 srcproc, grid->nprow);
            const int tag = loc_new_row/block_size;
            MPI_Request sreq;
            //MPI_Send(buf, block_size, datatype, dest_rank, tag, grid->all_comm);
            MPI_Isend(origmat->dA + loc_orig_row + origmat->ld*loc_b_orig_col, block_size,
                      datatype, dest_rank, tag, grid->all_comm, &sreq);

            //hipHostFree(buf);
            //free(buf);
        }
    }

    if(grid->mycol == new_b_proc_col) {
        if(loc_b_new_col != mat->nq-1) {
            throw std::runtime_error("Inconsistent b location in split matrix");
        }
        for(int loc_row = 0; loc_row < mat->mp; loc_row += block_size) {
            //printf("loc_row = %d, loc_col = %d, mat->mp = %d.\n", loc_row, loc_b_new_col, mat->mp);
            const int gl_row = HPL_indxl2g(loc_row, block_size, block_size, grid->myrow,
                                           srcproc, grid->nprow);
            const int old_proc_row = HPL_indxg2p(gl_row, origmat->nb, origmat->nb, srcproc,
                                                 grid->nprow);
            const int loc_old_row = HPL_indxg2l(gl_row, origmat->nb, origmat->nb, srcproc,
                                                grid->nprow);
            
            //scalar *buf{};
            //hipHostMalloc(&buf, block_size*sizeof(scalar));
            //buf = static_cast<scalar*>(malloc(block_size*sizeof(scalar)));

            const int source_rank = grid->order == HPL_ROW_MAJOR ?
                old_proc_row * grid->npcol + orig_b_proc_col :
                old_proc_row + orig_b_proc_col * grid->nprow;
            const int tag = loc_row/block_size;
            //MPI_Recv(buf, block_size, datatype, source_rank, tag, grid->all_comm, MPI_STATUS_IGNORE);
            MPI_Request req;
            MPI_Irecv(mat->dA + loc_row + mat->ld*loc_b_new_col, block_size, datatype, source_rank,
                      tag, grid->all_comm, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            
            //hipMemcpy(mat->dA + loc_row + mat->ld*loc_b_new_col, buf, block_size*sizeof(scalar),
            //          hipMemcpyHostToDevice);
            //hipHostFree(buf);
            //free(buf);

        }
    }

    if(grid->myrow == 0 && grid->mycol == 0) {
        printf("Completed splitting b vector.\n"); fflush(stdout);
    }
}
