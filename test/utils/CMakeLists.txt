
add_library(utils test_config.cpp)
target_link_libraries(utils PRIVATE MPI::MPI_CXX)
