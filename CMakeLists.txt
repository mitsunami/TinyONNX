cmake_minimum_required(VERSION 3.16)
project(TinyONNX LANGUAGES CXX)

# Explicitly force Clang compiler
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-14" CACHE STRING "C++ compiler" FORCE)
set(CMAKE_C_COMPILER "/usr/bin/clang-14" CACHE STRING "C compiler" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# XNNPACK
FetchContent_Declare(
  XNNPACK
  GIT_REPOSITORY https://github.com/google/XNNPACK.git
  GIT_TAG test_647082366
)
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ASSEMBLY OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM ON CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_X86 ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(XNNPACK)

# ONNX
find_package(Protobuf REQUIRED)
set(ONNX_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto)
set(GENERATED_SRC ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)
set(GENERATED_HDR ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)
add_custom_command(
  OUTPUT ${GENERATED_SRC} ${GENERATED_HDR}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/onnx
          --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
          ${ONNX_PROTO}
  DEPENDS ${ONNX_PROTO}
  COMMENT "Generating onnx.pb.h and onnx.pb.cc with system protoc"
)
add_library(onnx_proto STATIC ${GENERATED_SRC})
target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

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
    src/graph.cpp
)
target_include_directories(TinyONNX_lib PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
include_directories(
    include
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
    XNNPACK
    pthreadpool
    OpenMP::OpenMP_CXX
)

# Main executable
add_executable(TinyONNX src/main.cpp)
target_link_libraries(TinyONNX TinyONNX_lib)

target_compile_options(TinyONNX PRIVATE -Wall -Wextra -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TinyONNX PRIVATE -march=armv8-a+simd)
endif()


# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # optional for MSVC
FetchContent_MakeAvailable(googletest)

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
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_executable(TinyONNX_benchmarks
    benchmarks/benchmark_main.cpp
    benchmarks/simple_model_bench.cpp
    benchmarks/matmul_bench.cpp
    benchmarks/conv2d_bench.cpp
)
target_link_libraries(TinyONNX_benchmarks
    benchmark::benchmark
    TinyONNX_lib
)
