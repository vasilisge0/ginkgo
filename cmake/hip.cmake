# We keep using NVCC/HCC for consistency with previous releases even if AMD
# updated everything to use NVIDIA/AMD in ROCM 4.1
set(GINKGO_HIP_PLATFORM_NVCC 0)
set(GINKGO_HIP_PLATFORM_HCC 0)

if(DEFINED ENV{HIP_PLATFORM})
    set(GINKGO_HIP_PLATFORM "$ENV{HIP_PLATFORM}")
elseif(GINKGO_HIPCONFIG_PATH)
    execute_process(COMMAND ${GINKGO_HIPCONFIG_PATH}
        --platform OUTPUT_VARIABLE GINKGO_HIP_PLATFORM)
else()
    message(FATAL_ERROR "No platform could be found for HIP. "
            "Set and export the environment variable HIP_PLATFORM.")
endif()
message(STATUS "HIP platform set to ${GINKGO_HIP_PLATFORM}")
set(HIP_PLATFORM_AMD_REGEX "hcc|amd")
set(HIP_PLATFORM_NVIDIA_REGEX "nvcc|nvidia")

if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    set(GINKGO_HIP_PLATFORM_HCC 1)
elseif (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
    set(GINKGO_HIP_PLATFORM_NVCC 1)
endif()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()

if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}"
    AND GINKGO_BUILD_CUDA AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.2)
    message(FATAL_ERROR "Ginkgo HIP backend requires CUDA >= 9.2.")
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HIPBLAS_PATH)
    if(DEFINED ENV{HIPBLAS_PATH})
        set(HIPBLAS_PATH $ENV{HIPBLAS_PATH} CACHE PATH "Path to which HIPBLAS has been installed")
    else()
        set(HIPBLAS_PATH "${ROCM_PATH}/hipblas" CACHE PATH "Path to which HIPBLAS has been installed")
    endif()
endif()

if(NOT DEFINED HIPFFT_PATH)
    if(DEFINED ENV{HIPFFT_PATH})
        set(HIPFFT_PATH $ENV{HIPFFT_PATH} CACHE PATH "Path to which HIPFFT has been installed")
    else()
        set(HIPFFT_PATH "${ROCM_PATH}/hipfft" CACHE PATH "Path to which HIPFFT has been installed")
    endif()
endif()

if(NOT DEFINED HIPRAND_PATH)
    if(DEFINED ENV{HIPRAND_PATH})
        set(HIPRAND_PATH $ENV{HIPRAND_PATH} CACHE PATH "Path to which HIPRAND has been installed")
    else()
        set(HIPRAND_PATH "${ROCM_PATH}/hiprand" CACHE PATH "Path to which HIPRAND has been installed")
    endif()
endif()

if(NOT DEFINED ROCRAND_PATH)
    if(DEFINED ENV{ROCRAND_PATH})
        set(ROCRAND_PATH $ENV{ROCRAND_PATH} CACHE PATH "Path to which ROCRAND has been installed")
    else()
        set(ROCRAND_PATH "${ROCM_PATH}/rocrand" CACHE PATH "Path to which ROCRAND has been installed")
    endif()
endif()

if(NOT DEFINED HIPSPARSE_PATH)
    if(DEFINED ENV{HIPSPARSE_PATH})
        set(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH} CACHE PATH "Path to which HIPSPARSE has been installed")
    else()
        set(HIPSPARSE_PATH "${ROCM_PATH}/hipsparse" CACHE PATH "Path to which HIPSPARSE has been installed")
    endif()
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

# Find HIPCC_CMAKE_LINKER_HELPER executable
find_program(
    HIP_HIPCC_CMAKE_LINKER_HELPER
    NAMES hipcc_cmake_linker_helper
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
if(NOT HIP_HIPCC_CMAKE_LINKER_HELPER)
    # Now search in default paths
    find_program(HIP_HIPCC_CMAKE_LINKER_HELPER hipcc_cmake_linker_helper)
endif()

find_program(
    HIP_HIPCONFIG_EXECUTABLE
    NAMES hipconfig
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
if(NOT HIP_HIPCONFIG_EXECUTABLE)
    # Now search in default paths
    find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
endif()

execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
            OUTPUT_VARIABLE GINKGO_HIP_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            )

if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
    # ensure ENV{CUDA_PATH} is set by the user
    if (NOT DEFINED ENV{CUDA_PATH})
        find_path(GINKGO_HIP_DEFAULT_CUDA_PATH "cuda.h" PATH /usr/local/cuda/include NO_DEFAULT_PATH)
        if (NOT GINKGO_HIP_DEFAULT_CUDA_PATH)
            message(FATAL_ERROR "HIP nvidia backend was requested but CUDA could not be "
                "located. Set and export the environment variable CUDA_PATH.")
         endif()
     endif()
endif()

## Setup all CMAKE variables to find HIP and its dependencies
list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")
if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake")
endif()
list(APPEND CMAKE_PREFIX_PATH
    "${HIPBLAS_PATH}/lib/cmake"
    "${HIPFFT_PATH}/lib/cmake"
    "${HIPRAND_PATH}/lib/cmake"
    "${HIPSPARSE_PATH}/lib/cmake"
    "${ROCRAND_PATH}/lib/cmake"
)

# NOTE: without this, HIP jacobi build takes a *very* long time. The reason for
# that is that these variables are seemingly empty by default, thus there is no
# proper optimization applied to the HIP builds otherwise.
set(HIP_HIPCC_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Flags used by the HIPCC compiler during DEBUG builds")
set(HIP_HIPCC_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}" CACHE STRING "Flags used by the HIPCC compiler during MINSIZEREL builds")
set(HIP_HIPCC_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "Flags used by the HIPCC compiler during RELEASE builds")
set(HIP_HIPCC_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the HIPCC compiler during RELWITHDEBINFO builds")

find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipfft) # optional dependency
find_package(hiprand REQUIRED)
find_package(hipsparse REQUIRED)
# At the moment, for hiprand to work also rocrand is required.
find_package(rocrand REQUIRED)
find_path(GINKGO_HIP_THRUST_PATH "thrust/complex.h"
    PATHS "${HIP_PATH}/../include"
    ENV HIP_THRUST_PATH)
if (NOT GINKGO_HIP_THRUST_PATH)
    message(FATAL_ERROR "Could not find the ROCm header thrust/complex.h which is required by Ginkgo HIP.")
endif()
