cmake_minimum_required(VERSION 3.13.4)
project(DoraMapperPasses)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM definitions and include directories
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Build the pass as a shared library
add_library(llvm_passes MODULE
    llvm_passes.cpp
)

# Set output directory to current directory
set_target_properties(llvm_passes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
)

# Link against LLVM
target_link_libraries(llvm_passes PRIVATE LLVM)

# Platform-specific settings
if(APPLE)
    set_target_properties(llvm_passes PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif()
