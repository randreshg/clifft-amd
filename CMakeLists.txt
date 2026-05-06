cmake_minimum_required(VERSION 3.24)
project(clifft_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLIFFT_CUDA_ENABLE_CUDA "Build the native CUDA sampler" OFF)
set(CLIFFT_SOURCE_DIR "./third_party/clifft" CACHE PATH "Path to the vendored Clifft source tree")
set(CLIFFT_MAX_QUBITS "128" CACHE STRING "Maximum supported qubits for vendored Clifft core")

if(CLIFFT_CUDA_ENABLE_CUDA)
    add_library(clifft_cuda STATIC src/cuda_sampler.cu)
else()
    add_library(clifft_cuda STATIC src/cuda_sampler.cpp)
endif()
target_include_directories(clifft_cuda PUBLIC include)

if(CLIFFT_SOURCE_DIR)
    get_filename_component(CLIFFT_SOURCE_DIR_ABS "${CLIFFT_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(FETCHCONTENT_SOURCE_DIR_STIM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stim" CACHE PATH "Path to vendored Stim source" FORCE)
    set(FETCHCONTENT_SOURCE_DIR_FAST_FLOAT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float" CACHE PATH "Path to vendored fast_float source" FORCE)
    set(CLIFFT_VERSION "0.1.0-cuda" CACHE STRING "Version string for vendored Clifft core")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated/clifft/util)
    configure_file(
        ${CLIFFT_SOURCE_DIR_ABS}/src/clifft/util/version.h.in
        ${CMAKE_BINARY_DIR}/generated/clifft/util/version.h
        @ONLY
    )
    include(${CLIFFT_SOURCE_DIR_ABS}/cmake/FetchStim.cmake)
    include(${CLIFFT_SOURCE_DIR_ABS}/cmake/FetchFastFloat.cmake)
    add_subdirectory(${CLIFFT_SOURCE_DIR_ABS}/src/clifft ${CMAKE_BINARY_DIR}/clifft_core)
    target_link_libraries(clifft_cuda PUBLIC clifft_core)
endif()

if(CLIFFT_CUDA_ENABLE_CUDA)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(clifft_cuda PRIVATE CUDA::cudart)
    target_compile_features(clifft_cuda PUBLIC cxx_std_20)
endif()

if(CLIFFT_SOURCE_DIR AND CLIFFT_CUDA_ENABLE_CUDA)
    add_executable(run_msc_cuda tools/run_msc_cuda.cpp)
    target_link_libraries(run_msc_cuda PRIVATE clifft_cuda clifft_core CUDA::cudart)
endif()
