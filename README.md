# TinyONNX

A lightweight ONNX Runtime

## Features
- Ultra-lightweight ONNX inference engine
- Optimized for Arm CPUs using SIMD (planned)
- Minimal dependency, small memory footprint

## Quickstart
```bash
mkdir build && cd build
cmake ..
make
./TinyONNX ../models/example.onnx
```
