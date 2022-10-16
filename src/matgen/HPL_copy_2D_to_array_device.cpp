#include "hpl.hpp"

template <typename T>
__global__
void copy_2d_to_array(T *const dest, const int src_stride, const int nrows, const int ncols,
                      const T *const src)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= nrows * ncols) {
        return;
    }
    const auto col = tid / nrows;
    const auto row = tid % nrows;
    dest[tid] = src[col*src_stride + row];
}

template <typename T>
void HPL_device_copy_2d_to_array(T *const dest, const int src_stride,
    const int nrows, const int ncols, const T *const src)
{
    const int nts = nrows * ncols;
    constexpr int block_size = 512;
    const int nblocks = (nts - 1) / block_size + 1;
    copy_2d_to_array<<<nblocks, block_size>>>(dest, src_stride, nrows, ncols, src);
}

template void HPL_device_copy_2d_to_array<double>(double *const dest, const int src_stride,
    const int nrows, const int ncols, const double *const src);
