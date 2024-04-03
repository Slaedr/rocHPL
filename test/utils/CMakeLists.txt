
add_library(utils test_config.cpp test_utils.cpp)
target_link_libraries(utils PRIVATE MPI::MPI_CXX)
