cmake_minimum_required(VERSION 3.16)
project(TinyONNX LANGUAGES CXX)

# Explicitly force Clang compiler
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

set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY /usr/lib/llvm-14/lib/libomp.so)
find_package(OpenMP REQUIRED)

# Create TinyONNX core as a static library
add_library(TinyONNX_lib STATIC
    src/onnx_loader.cpp
    src/execution_engine.cpp
    src/tensor.cpp
    src/operators.cpp
    src/conv2d.cpp
    src/onnx_utils.cpp
    src/graph_utils.cpp
)

option(ENABLE_TIMING "Enable profiling timers" OFF)
if (ENABLE_TIMING)
    add_definitions(-DENABLE_TIMING)
endif()

option(ENABLE_MEM_USAGE "Enable profiling memory usage" OFF)
if (ENABLE_MEM_USAGE)
    add_definitions(-DENABLE_MEM_USAGE)
endif()

target_link_libraries(TinyONNX_lib
    onnx_proto
    protobuf::libprotobuf
    OpenMP::OpenMP_CXX
)

# Main executable (now uses library)
add_executable(TinyONNX src/main.cpp)
target_link_libraries(TinyONNX TinyONNX_lib)

target_compile_definitions(TinyONNX PRIVATE ONNX_ML=1)
target_compile_options(TinyONNX PRIVATE -Wall -Wextra -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TinyONNX PRIVATE -march=armv8-a+simd)
endif()


# GoogleTest
add_subdirectory(third_party/googletest)
enable_testing()

add_executable(TinyONNX_tests
    tests/test_main.cpp
    tests/test_graph_parsing.cpp
    tests/test_simple_model.cpp
    tests/test_conv2d.cpp
    tests/test_matmul.cpp
    tests/test_add.cpp
    tests/test_relu.cpp
    tests/test_softmax.cpp
    tests/test_batchnorm.cpp
    tests/test_global_avgpool.cpp
    tests/test_reshape.cpp
    tests/test_toposort.cpp
)

target_link_libraries(TinyONNX_tests
    gtest
    gtest_main
    TinyONNX_lib
)

include(GoogleTest)
gtest_discover_tests(TinyONNX_tests)


# Benchmarks
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
add_subdirectory(third_party/benchmark)

add_executable(TinyONNX_benchmarks
    benchmarks/benchmark_main.cpp
    benchmarks/simple_model_bench.cpp
    benchmarks/matmul_bench.cpp
    benchmarks/conv2d_bench.cpp
)
target_link_libraries(TinyONNX_benchmarks
    benchmark
    TinyONNX_lib
)
