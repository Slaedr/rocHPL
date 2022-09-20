
#include "hpl.hpp"
#include "hpl_pmatgen.hpp"

template <typename scalar>
__global__
void copy2d(size_t source_ld, int nrows, int ncols, const scalar *source,
            size_t dest_ld, scalar *const dest)
{
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = flat_idx % nrows;
    const int j = flat_idx / nrows;
    if(i >= nrows || j >= ncols) {
        return;
    }

    dest[i + j*dest_ld] = source[i + j*source_ld];
}


template <typename scalar>
void device_copy_2d_block(size_t source_ld, int nrows, int ncols, const scalar *source,
                          size_t dest_ld, scalar *const dest)
{
    constexpr int block_size = 1024;
    const int nblocks = (nrows * ncols - 1) / block_size + 1;
    hipLaunchKernelGGL(copy2d, nblocks, block_size, 0, 0, source_ld, nrows, ncols, source,
        dest_ld, dest);
}

template void device_copy_2d_block(size_t source_ld, int nrows, int ncols,
        const float *source, size_t dest_ld, float *const dest);
template void device_copy_2d_block(size_t source_ld, int nrows, int ncols,
        const double *source, size_t dest_ld, double *const dest);

