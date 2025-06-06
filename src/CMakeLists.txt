# Source files for the library
set(LIB_SOURCES
    tensor.cpp
    tensor_ops.cpp
    tensor_ops_impl.cpp
    model_loader.cpp
    transformer.cpp
    tokenizer.cpp
    kv_cache.cpp
)

# Add the main library
add_library(lightgpt ${LIB_SOURCES})

# Set target properties
target_include_directories(lightgpt
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lightgpt
    PRIVATE 
        Threads::Threads
        nlohmann_json::nlohmann_json
)

# Add the main executable
add_executable(lightgpt-cli
    main.cpp
)

target_link_libraries(lightgpt-cli
    PRIVATE lightgpt
)

# Set output directory
set_target_properties(lightgpt-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    OUTPUT_NAME lightgpt
)

# Installation rules
install(TARGETS lightgpt lightgpt-cli
    EXPORT lightgptTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/lightgpt
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
)
