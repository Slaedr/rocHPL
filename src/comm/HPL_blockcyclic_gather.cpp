
#include "hpl.hpp"

int HPL_1D_blockcyclic_gather(const void *const sendbuf, const int sendcount,
    void *const recvbuf, const int recvcount, const int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    using scalar = double;
    
    return 0;
}
