
#include <unistd.h>
#include <stdlib.h>

#include "hpl_misc.hpp"

int HPL_malloc(void** ptr, const size_t bytes) {

  //int mycol, myrow, npcol, nprow;
  //(void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  unsigned long pg_size = sysconf(_SC_PAGESIZE);
  int           err     = posix_memalign(ptr, pg_size, bytes);

  /*Check workspace allocation is valid*/
  if(err != 0) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}
