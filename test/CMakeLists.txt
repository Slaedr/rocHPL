include_directories(${hpl_include_dirs})
include_directories("${CMAKE_SOURCE_DIR}/test")

# Build testing infrastructure (and their tests) first
add_subdirectory(gather)
add_library(test_utils test_utils.cpp)

# Build HPL tests
add_executable(grid_tests test_grid.cpp)
target_link_libraries(grid_tests PRIVATE test_utils rochpl hip::host)
