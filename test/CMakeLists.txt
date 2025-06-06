# Enable testing
enable_testing()

# Use FetchContent to get GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Only build gtest and gtest_main, not the whole gtest
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Add test executables
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor
    PRIVATE
        gtest
        gtest_main
        lightgpt
)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer
    PRIVATE
        gtest
        gtest_main
        lightgpt
)

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer
    PRIVATE
        gtest
        gtest_main
        lightgpt
)

# Add test_model_loader executable
add_executable(test_model_loader test_model_loader.cpp)
target_link_libraries(test_model_loader
    PRIVATE
        gtest
        gtest_main
        lightgpt
)

# Add tests
add_test(NAME test_tensor COMMAND test_tensor)
add_test(NAME test_tokenizer COMMAND test_tokenizer)
add_test(NAME test_transformer COMMAND test_transformer)
add_test(NAME test_model_loader COMMAND test_model_loader)
