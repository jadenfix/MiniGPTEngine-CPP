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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Enable SIMD optimizations
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    add_compile_definitions(USE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# Project directories
set(LIGHTGPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIGHTGPT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIGHTGPT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Include directories
include_directories(${LIGHTGPT_INCLUDE_DIR})

# Core library sources
set(LIGHTGPT_SOURCES
    ${LIGHTGPT_SRC_DIR}/model_loader.cpp
    ${LIGHTGPT_SRC_DIR}/tensor.cpp
    ${LIGHTGPT_SRC_DIR}/tensor_ops.cpp
    ${LIGHTGPT_SRC_DIR}/attention.cpp
    ${LIGHTGPT_SRC_DIR}/feedforward.cpp
    ${LIGHTGPT_SRC_DIR}/transformer_block.cpp
    ${LIGHTGPT_SRC_DIR}/inference_engine.cpp
    ${LIGHTGPT_SRC_DIR}/tokenizer.cpp
    ${LIGHTGPT_SRC_DIR}/utils.cpp
)

# Core library headers
set(LIGHTGPT_HEADERS
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/model_loader.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/tensor.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/tensor_ops.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/attention.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/feedforward.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/transformer_block.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/inference_engine.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/tokenizer.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/utils.hpp
    ${LIGHTGPT_INCLUDE_DIR}/lightgpt/types.hpp
)

# Create static library
add_library(lightgpt_core STATIC ${LIGHTGPT_SOURCES} ${LIGHTGPT_HEADERS})
target_include_directories(lightgpt_core PUBLIC ${LIGHTGPT_INCLUDE_DIR})

# Main CLI executable
add_executable(lightgpt ${LIGHTGPT_SRC_DIR}/main.cpp)
target_link_libraries(lightgpt lightgpt_core)

# Testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Find or fetch GoogleTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50f33f9142fd2d6a04e5e5bb8244cd.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Test sources
    set(TEST_SOURCES
        ${LIGHTGPT_TEST_DIR}/test_model_loader.cpp
        ${LIGHTGPT_TEST_DIR}/test_tensor.cpp
        ${LIGHTGPT_TEST_DIR}/test_tensor_ops.cpp
        ${LIGHTGPT_TEST_DIR}/test_attention.cpp
        ${LIGHTGPT_TEST_DIR}/test_transformer_block.cpp
        ${LIGHTGPT_TEST_DIR}/test_inference_engine.cpp
        ${LIGHTGPT_TEST_DIR}/test_tokenizer.cpp
    )
    
    # Create test executables
    foreach(test_file ${TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} lightgpt_core gtest gtest_main)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
    
    # Main test runner
    add_executable(run_all_tests ${TEST_SOURCES})
    target_link_libraries(run_all_tests lightgpt_core gtest gtest_main)
endif()

# Benchmarking
add_executable(benchmark ${LIGHTGPT_SRC_DIR}/benchmark.cpp)
target_link_libraries(benchmark lightgpt_core)

# Installation
install(TARGETS lightgpt lightgpt_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${LIGHTGPT_INCLUDE_DIR}/lightgpt DESTINATION include)
