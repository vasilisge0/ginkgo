function(ginkgo_build_test_name test_name target_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    set(${target_name} ${TEST_TARGET_NAME} PARENT_SCOPE)
endfunction()

## Set up shared target properties and handle ADDITIONAL_LIBRARIES/ADDITIONAL_INCLUDES
## `MPI_SIZE size` causes the tests to be run with `size` MPI processes.
function(ginkgo_set_test_target_properties test_target_name)
    cmake_parse_arguments(PARSE_ARGV 1 set_properties "" "MPI_SIZE" "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_INCLUDES")
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_COMPILING_DPCPP_TEST AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${test_target_name} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    if (set_properties_MPI_SIZE)
        set(gtest_main gtest_mpi_main MPI::MPI_CXX)
    else()
        set(gtest_main GTest::Main)
    endif()
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
    target_include_directories(${test_target_name} PRIVATE ${Ginkgo_BINARY_DIR} ${set_properties_ADDITIONAL_INCLUDES})
    target_link_libraries(${test_target_name} PRIVATE ginkgo ${gtest_main} GTest::GTest ${set_properties_ADDITIONAL_LIBRARIES})
endfunction()

## Adds a test to the list executed by ctest and sets its output binary name
function(ginkgo_add_test test_name test_target_name)
    cmake_parse_arguments(PARSE_ARGV 2 add_test "" "MPI_SIZE" "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_INCLUDES")
    file(RELATIVE_PATH REL_BINARY_DIR ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${test_name})
    if (add_test_MPI_SIZE)
        if("${GINKGO_MPI_EXEC_SUFFIX}" MATCHES ".openmpi" AND MPI_RUN_AS_ROOT)
            set(OPENMPI_RUN_AS_ROOT_FLAG "--allow-run-as-root")
        else()
            set(OPENMPI_RUN_AS_ROOT_FLAG "")
        endif()
        add_test(NAME ${REL_BINARY_DIR}/${test_name}
                 COMMAND
                     ${MPIEXEC_EXECUTABLE}
                     ${MPIEXEC_NUMPROC_FLAG}
                     ${add_test_MPI_SIZE}
                     ${OPENMPI_RUN_AS_ROOT_FLAG}
                     "$<TARGET_FILE:${test_target_name}>"
                 WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    else()
        add_test(NAME ${REL_BINARY_DIR}/${test_name}
                 COMMAND ${test_target_name}
                 WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    endif()
endfunction()

## Normal test
function(ginkgo_create_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_link_libraries(${test_target_name} PRIVATE ${create_test_ADDITIONAL_LIBRARIES})
    ginkgo_set_test_target_properties(${test_target_name} ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN})
endfunction(ginkgo_create_test)

## Test compiled with dpcpp
function(ginkgo_create_dpcpp_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.dp.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_17)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_DPCPP_FLAGS})
    target_link_options(${test_target_name} PRIVATE -fsycl-device-code-split=per_kernel)
    ginkgo_set_test_target_properties(${test_target_name} ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN})
    # Note: MKL_ENV is empty on linux. Maybe need to apply MKL_ENV to all test.
    if (MKL_ENV)
        set_tests_properties(${test_target_name} PROPERTIES ENVIRONMENT "${MKL_ENV}")
    endif()
endfunction(ginkgo_create_dpcpp_test)

function(ginkgo_create_cuda_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    ginkgo_create_cuda_test_internal(${test_name} ${test_name}.cu ${test_target_name} ${ARGN})
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_cuda_test_internal test_name filename test_target_name)
    add_executable(${test_target_name} ${filename})
    target_compile_definitions(${test_target_name} PRIVATE GKO_COMPILING_CUDA)
    target_compile_options(${test_target_name}
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>)
    if(MSVC)
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    endif()
    # we handle CUDA architecture flags for now, disable CMake handling
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_target_properties(${test_target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
    ginkgo_set_test_target_properties(${test_target_name} ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN})
endfunction(ginkgo_create_cuda_test_internal)

function(ginkgo_create_hip_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    ginkgo_create_hip_test_internal(${test_name} ${test_name}.hip.cpp ${test_target_name} "" ${ARGN})
endfunction(ginkgo_create_hip_test)

function(ginkgo_create_hip_test_internal test_name filename test_target_name additional_flags)
    set_source_files_properties(${filename} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    set(GINKGO_TEST_HIP_DEFINES -DGKO_COMPILING_HIP ${additional_flags})
    if (GINKGO_FAST_TESTS)
        list(APPEND GINKGO_TEST_HIP_DEFINES -DGINKGO_FAST_TESTS)
    endif()

    # NOTE: With how HIP works, passing the flags `HIPCC_OPTIONS` etc. here
    # creates a redefinition of all flags. This creates some issues with `nvcc`,
    # but `clang` seems fine with the redefinitions.
    if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
        hip_add_executable(${test_target_name} ${filename}
            # If `FindHIP.cmake`, namely `HIP_PARSE_HIPCC_OPTIONS` macro and
            # call gets fixed, uncomment this.
            HIPCC_OPTIONS ${GINKGO_TEST_HIP_DEFINES} # ${GINKGO_HIPCC_OPTIONS}
            # NVCC_OPTIONS  ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_NVCC_OPTIONS}
            # CLANG_OPTIONS ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_CLANG_OPTIONS}
            --expt-relaxed-constexpr --expt-extended-lambda
            )
    else() # hcc/clang
        hip_add_executable(${test_target_name} ${filename}
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} ${GINKGO_TEST_HIP_DEFINES}
            NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
            CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
            )
    endif()

    # Let's use a normal compiler for linking
    set_target_properties(${test_target_name} PROPERTIES LINKER_LANGUAGE CXX)

    target_include_directories(${test_target_name}
        PRIVATE
        # Only `math` requires it so far, but it's much easier
        # to put these this way.
        ${GINKGO_HIP_THRUST_PATH}
        # Only `exception_helpers` requires these so far, but it's much easier
        # to put these this way.
        ${HIPBLAS_INCLUDE_DIRS}
        ${HIPFFT_INCLUDE_DIRS}
        ${hiprand_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        )
    ginkgo_set_test_target_properties(${test_target_name} ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN})
endfunction(ginkgo_create_hip_test_internal)

function(ginkgo_create_common_test test_name)
    if(GINKGO_BUILD_OMP)
        ginkgo_create_common_test_internal(${test_name} OmpExecutor omp ${ARGN})
    endif()
    if(GINKGO_BUILD_HIP)
        ginkgo_create_common_test_internal(${test_name} HipExecutor hip ${ARGN})
    endif()
    if(GINKGO_BUILD_CUDA)
        ginkgo_create_common_test_internal(${test_name} CudaExecutor cuda ${ARGN})
    endif()
    if(GINKGO_BUILD_DPCPP)
        ginkgo_create_common_test_internal(${test_name} DpcppExecutor dpcpp ${ARGN})
    endif()
endfunction(ginkgo_create_common_test)

function(ginkgo_create_common_test_internal test_name exec_type exec)
    cmake_parse_arguments(PARSE_ARGV 1 common_test "" "MPI_SIZE" "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_INCLUDES")
    if(exec IN_LIST common_test_DISABLE_EXECUTORS)
        return()
    endif()
    ginkgo_build_test_name(${test_name} test_target_name)
    string(TOUPPER ${exec} exec_upper)
    # set up actual test
    set(test_target_name ${test_target_name}_${exec})
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_definitions(${test_target_name} PRIVATE EXEC_TYPE=${exec_type} EXEC_NAMESPACE=${exec} GKO_COMPILING_${exec_upper})
    target_link_libraries(${test_target_name} PRIVATE ${common_test_ADDITIONAL_LIBRARIES})
    # use float for DPC++ if necessary
    if((exec STREQUAL "dpcpp") AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_COMMON_SINGLE_MODE=1)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    ginkgo_set_test_target_properties(${test_target_name} ${ARGN})
    ginkgo_add_test(${test_name}_${exec} ${test_target_name} ${ARGN})
endfunction(ginkgo_create_common_test_internal)

function(ginkgo_create_common_device_test test_name)
    cmake_parse_arguments(PARSE_ARGV 1 common_device_test "" "MPI_SIZE" "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_INCLUDES")
    ginkgo_build_test_name(${test_name} test_target_name)
    if(GINKGO_BUILD_DPCPP)
        ginkgo_create_common_test_internal(${test_name} DpcppExecutor dpcpp ${ARGN})
        target_compile_features(${test_target_name}_dpcpp PRIVATE cxx_std_17)
        target_link_options(${test_target_name}_dpcpp PRIVATE -fsycl-device-lib=all)
    endif()
    if(GINKGO_BUILD_OMP)
        ginkgo_create_common_test_internal(${test_name} OmpExecutor omp ${ARGN})
        target_link_libraries(${test_target_name}_omp PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(GINKGO_BUILD_CUDA)
        # need to make a separate file for this, since we can't set conflicting properties on the same file
        configure_file(${test_name}.cpp ${test_name}.cu COPYONLY)
        ginkgo_create_cuda_test_internal(${test_name}_cuda ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cu ${test_target_name}_cuda ${ARGN})
        target_compile_definitions(${test_target_name}_cuda PRIVATE EXEC_TYPE=CudaExecutor EXEC_NAMESPACE=cuda)
    endif()
    if(GINKGO_BUILD_HIP)
        # need to make a separate file for this, since we can't set conflicting properties on the same file
        configure_file(${test_name}.cpp ${test_name}.hip.cpp COPYONLY)
        ginkgo_create_hip_test_internal(${test_name}_hip ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.hip.cpp ${test_target_name}_hip "-std=c++14;-DEXEC_TYPE=HipExecutor;-DEXEC_NAMESPACE=hip" ${ARGN})
    endif()
endfunction(ginkgo_create_common_device_test)

function(ginkgo_create_common_and_reference_test test_name)
    ginkgo_create_common_test(${test_name} ${ARGN})
    ginkgo_create_common_test_internal(${test_name} ReferenceExecutor reference REFERENCE ${ARGN})
endfunction(ginkgo_create_common_and_reference_test)
