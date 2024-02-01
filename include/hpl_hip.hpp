#ifndef ROCHPL_HIP_HPP_
#define ROCHPL_HIP_HPP_

// NC: hipcc in ROCm 3.7 complains if __HIP_PLATFORM_HCC__ is defined in the
// compile line
#ifdef __HIPCC__
#ifdef __HIP_PLATFORM_HCC__
#undef __HIP_PLATFORM_HCC__
#endif
#endif

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <roctracer.h>
#include <roctx.h>

#include "hpl_pgesv_types.hpp"

extern hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
extern hipEvent_t swapUCopyEvent[HPL_N_UPD], swapWCopyEvent[HPL_N_UPD];
extern hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];

extern rocblas_handle handle;
extern hipStream_t    computeStream;
extern hipStream_t    dataStream;

#endif
