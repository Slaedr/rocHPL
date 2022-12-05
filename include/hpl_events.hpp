#ifndef ROCHPL_EVENTS_HPP_
#define ROCHPL_EVENTS_HPP_

#include <hip/hip_runtime_api.h>

extern hipEvent_t swapStartEvent[HPL_N_UPD];
extern hipEvent_t update[HPL_N_UPD];
extern hipEvent_t swapUCopyEvent[HPL_N_UPD], swapWCopyEvent[HPL_N_UPD];
extern hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];

#endif
