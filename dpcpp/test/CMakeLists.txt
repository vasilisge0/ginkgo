include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)
set(GINKGO_COMPILING_DPCPP_TEST ON)

add_subdirectory(base)
add_subdirectory(components)
add_subdirectory(matrix)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(stop)
