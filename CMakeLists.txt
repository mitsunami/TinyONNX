cmake_minimum_required(VERSION 3.16)
project(TinyONNX LANGUAGES CXX)

# Explicitly force Clang compiler
#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE STRING "C compiler" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    include
    ${CMAKE_SOURCE_DIR}/third_party/onnx
    ${CMAKE_SOURCE_DIR}/third_party/onnx/build
)

# ONNX submodule integration
set(ONNX_USE_PROTOBUF_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/onnx)

find_package(Protobuf REQUIRED)

set(SRC_FILES
    src/main.cpp
    src/onnx_loader.cpp
    src/execution_engine.cpp
    src/tensor.cpp
    src/operators.cpp
)
add_executable(TinyONNX ${SRC_FILES})

# Link libraries
target_link_libraries(TinyONNX
    onnx_proto
    protobuf::libprotobuf
)

target_compile_definitions(TinyONNX PRIVATE ONNX_ML=1)
target_compile_options(TinyONNX PRIVATE -Wall -Wextra -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TinyONNX PRIVATE -march=armv8-a+simd)
endif()
