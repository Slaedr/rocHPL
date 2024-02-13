add_executable(pfact_shared pfact_shared.cpp)
target_link_libraries(pfact_shared PRIVATE matutils matgen rochpl_host)
target_link_libraries(pfact_shared PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(pfact_dist pfact_dist.cpp)
target_link_libraries(pfact_dist PRIVATE matutils matgen utils rochpl_host)
target_link_libraries(pfact_dist PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/pfact_dist DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
