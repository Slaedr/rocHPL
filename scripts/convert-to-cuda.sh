#! /bin/bash

find . -name '*pp' -exec sed -i 's/hip/cuda/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_fill_\(\w*\)/CUBLAS_FILL_MODE_\U\1\E/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_diagonal_\(\.*\)/CUBLAS_DIAG_\U\1\E/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_side_\(\w*\)/CUBLAS_SIDE_\U\1\E/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_operation_\(\w\)\(.*\)/CUBLAS_OP_\u\1/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_get_stream/cublasGetStream/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_\(\w\)\(.*\)/cublas\u\1\2/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_handle/cublasHandle_t/g' {} \;
find . -name '*pp' -exec sed -i 's/roctxRangePush/nvtxRangePushA/g' {} \;
find . -name '*pp' -exec sed -i 's/roctx\.h/nvtx3\/nvToolsExt.h/g' {} \;
find . -name '*pp' -exec sed -i 's/roctx/nvtx/g' {} \;
#find . -name '*.cpp' -exec sed -i 's/cublas\(\w*\)_\(\w\)/cublas\1\u\2/g' {} \;

find . -name '*pp' -exec sed -i 's/cudaMallocHost/cudaHostMalloc/g' {} \;
