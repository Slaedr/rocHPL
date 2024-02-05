add_library(matgen HPL_dmatgen.cpp  HPL_jumpit.cpp  HPL_ladd.cpp  HPL_lmul.cpp  
    HPL_rand.cpp  HPL_setran.cpp  HPL_xjumpm.cpp)
target_link_libraries(matgen PUBLIC MPI::MPI_CXX)

