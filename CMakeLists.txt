cmake_minimum_required(VERSION 3.16)
project(lightgpt VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")

# Enable SIMD optimizations
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    add_compile_definitions(USE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    add_compile_definitions(USE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Project directories
set(LIGHTGPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Include directories
include_directories(${LIGHTGPT_INCLUDE_DIR})

# Create header-only interface library
add_library(lightgpt_headers INTERFACE)
target_include_directories(lightgpt_headers INTERFACE ${LIGHTGPT_INCLUDE_DIR})
target_compile_features(lightgpt_headers INTERFACE cxx_std_20)

# Set optimization flags for release
target_compile_options(lightgpt_headers INTERFACE
    $<$<CONFIG:Release>:-O3 -march=native -mavx2 -flto>
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lightgpt_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

# Advanced optimization test executable
add_executable(advanced_throughput_test
    advanced_throughput_test.cpp
)

target_link_libraries(advanced_throughput_test
    lightgpt_headers
)

target_compile_options(advanced_throughput_test PRIVATE
    -O3 -march=native -mavx2
    ${OpenMP_CXX_FLAGS}
)

# Comprehensive optimization test  
add_executable(test_optimizations
    test_optimizations.cpp
)

target_link_libraries(test_optimizations
    lightgpt_headers
)

target_compile_options(test_optimizations PRIVATE
    -O3 -march=native -mavx2
    ${OpenMP_CXX_FLAGS}
)

# Real performance benchmark
add_executable(real_performance_benchmark
    real_performance_benchmark.cpp
)

target_link_libraries(real_performance_benchmark
    lightgpt_headers
)

target_compile_options(real_performance_benchmark PRIVATE
    -O3 -march=native -mavx2
    ${OpenMP_CXX_FLAGS}
)

# Simple performance test
add_executable(simple_perf_test
    simple_perf_test.cpp
)

target_link_libraries(simple_perf_test
    lightgpt_headers
)

target_compile_options(simple_perf_test PRIVATE
    -O3 -march=native
    ${OpenMP_CXX_FLAGS}
)

# Comprehensive validation test
add_executable(comprehensive_validation
    comprehensive_validation.cpp
)

target_link_libraries(comprehensive_validation
    lightgpt_headers
)

target_compile_options(comprehensive_validation PRIVATE
    -O3 -march=native -mavx2
    ${OpenMP_CXX_FLAGS}
)

# Installation
install(DIRECTORY ${LIGHTGPT_INCLUDE_DIR}/lightgpt DESTINATION include)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 support: YES")
else()
    message(STATUS "AVX2 support: NO")
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support: YES")
else()
    message(STATUS "OpenMP support: NO")
endif()
