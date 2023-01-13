include_directories(${hpl_include_dirs})
include_directories("${CMAKE_SOURCE_DIR}/test")

hip_add_library(gather STATIC gather_matrix.cpp)
target_include_directories(gather PRIVATE ${CNPY_INCLUDE_DIR})
target_link_libraries(gather PUBLIC rochpl)
target_link_libraries(gather PUBLIC hip::host)

# Generate matrix files
message(STATUS "Running Python3 to generate matrix files for gather test")
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Tests
add_executable(test_gather test_gather.cpp)
target_link_libraries(test_gather gather test_utils)

add_test(NAME MatrixGather_1node
    COMMAND srun -n 8 test_gather -N 1200 -NB 80 -P 1 -Q 8 --matrix_dir ${CMAKE_CURRENT_BINARY_DIR} 
    --matrix_dir_type flat --reference_solution_path full_A.npy)

add_test(NAME MatrixGather_2node
    COMMAND srun -n 16 test_gather -N 1200 -NB 80 -P 2 -Q 8 --matrix_dir ${CMAKE_CURRENT_BINARY_DIR} 
    --matrix_dir_type flat --reference_solution_path full_A.npy)
