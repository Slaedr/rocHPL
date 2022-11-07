
#include "hpl.hpp"

int HPL_bcast_1ring_sync(double* SBUF, const int SCOUNT, const int ROOT, MPI_Comm COMM)
{
    int rank, size;
    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &size);
    
    if(size <= 1) {
        return (MPI_SUCCESS);
    }
    
    /*One ring exchange to rule them all*/
    MPI_Status myStatus;
    const int tag = rank;
    const int next = MModAdd1(rank, size);
    const int prev = MModSub1(rank, size);
    
    /*Mid point of message*/
    double* RBUF = SBUF;
    
    /*Shift to ROOT=0*/
    rank = MModSub(rank, ROOT, size);
    const int Nsend = (rank == size - 1) ? 0 : SCOUNT;
    const int Nrecv = (rank == 0) ? 0 : SCOUNT;
    
    /*Recv from left*/
    if(Nrecv > 0) {
        MPI_Recv(RBUF, Nrecv, MPI_DOUBLE, prev, prev, COMM, &myStatus);
    }
    
    /*Send to right if there is data present to send*/
    if(Nsend > 0) {
        MPI_Send(SBUF, Nsend, MPI_DOUBLE, next, tag, COMM);
    }
    
    return MPI_SUCCESS;
}
