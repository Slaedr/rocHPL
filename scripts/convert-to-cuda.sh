#! /bin/bash

find . -name '*pp' -exec sed -i 's/hip/cuda/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_fill_\(\w*\)/CUBLAS_FILL_MODE_\U\1\E/gp' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_diagonal_\(\p*\)/CUBLAS_DIAG_\U\1\E/gp' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_side_\(\p*\)/CUBLAS_SIDE_\U\1\E/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_operation_\(\w\)\(\p*\)/CUBLAS_OP_\u\1/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_get_stream/cublasGetStream/g' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_\(\w\)\(.*\)/cublas\u\1\2/gp' {} \;
find . -name '*pp' -exec sed -i 's/rocblas_handle/cublasHandle_t/gp' {} \;
find . -name '*pp' -exec sed -i 's/roctxRangePush/nvtxRangePushA/gp' {} \;
find . -name '*pp' -exec sed -i 's/roctx\.h/nvtx3\/nvToolsExt.h/gp' {} \;
find . -name '*pp' -exec sed -i 's/roctx/nvtx/gp' {} \;
#find . -name '*.cpp' -exec sed -i 's/cublas\(\w*\)_\(\w\)/cublas\1\u\2/g' {} \;
