add_executable(pfact_shared pfact_shared.cpp)
target_link_libraries(pfact_shared PRIVATE matgen rochpl_host)
target_link_libraries(pfact_shared PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(pfact_shared PUBLIC MPI::MPI_CXX)
