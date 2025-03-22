# TinyONNX

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/mitsunami/TinyONNX/actions/workflows/build.yml/badge.svg)](https://github.com/mitsunami/TinyONNX/actions/workflows/build.yml)

A lightweight ONNX Runtime

## Features
- Ultra-lightweight ONNX inference engine
- Optimized for Arm CPUs using SIMD (planned)
- Minimal dependency, small memory footprint

## Setup and Build Instructions

### Clone repository with submodules
```bash
git clone --recursive https://github.com/mitsunami/TinyONNX
cd TinyONNX
```

### Initialize submodules
```bash
git submodule update --init --recursive
```

### Build Project
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Dependencies
- Clang Compiler
- Protobuf
- ONNX (included as git submodule)

### Notes
- Ensure Clang is installed (`clang` and `clang++`).
- Ensure Protobuf libraries and headers are installed on your system.


