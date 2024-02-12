# For now, this test build does not depend on HIP and does not build HIP code.

add_compile_definitions(HPL_TEST_BUILD)

include_directories($<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

add_subdirectory(matgen)
add_subdirectory(utils)

enable_testing()

add_subdirectory(pfact)

