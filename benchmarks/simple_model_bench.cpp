#include <benchmark/benchmark.h>
#include "onnx_loader.h"
#include "execution_engine.h"

static void BM_SimpleModel(benchmark::State& state) {
    ONNXModel model;
    model.load("../models/simple_matmul_relu.onnx");

    ComputationGraph graph = model.parseGraph();

    Tensor input({1, 224});
    input.fillRandom();

    ExecutionEngine engine;

    for (auto _ : state) {
        engine.executeGraph(graph, input);
    }
}

BENCHMARK(BM_SimpleModel)->Iterations(50);
