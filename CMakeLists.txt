cmake_minimum_required(VERSION 3.16)
project(TinyONNX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
find_package(Protobuf REQUIRED)

add_executable(TinyONNX
    src/main.cpp
    src/onnx_loader.cpp
    src/execution_engine.cpp
    src/tensor.cpp
    src/operators.cpp
)

target_link_libraries(TinyONNX protobuf::libprotobuf)
