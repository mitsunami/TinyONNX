#include <gtest/gtest.h>
#include "onnx_loader.h"
#include "execution_engine.h"

TEST(IntegrationTest, SimpleModelMatMulReLU) {
    ONNXModel model;
    ASSERT_TRUE(model.load("../models/simple_matmul_relu.onnx"));

    ComputationGraph graph = model.parseGraph();

    Tensor input({1, 224});
    input.fillRandom();

    ExecutionEngine engine;
    ASSERT_NO_THROW(engine.executeGraph(graph, input));
}
