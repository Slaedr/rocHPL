
add_executable(trsm trsm.cpp)
target_link_libraries(trsm PRIVATE matutils matgen utils rochpl_host)
target_link_libraries(trsm PRIVATE OpenMP::OpenMP_CXX)
