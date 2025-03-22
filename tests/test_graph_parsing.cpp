#include <gtest/gtest.h>
#include "onnx_loader.h"

TEST(GraphParsingTest, ParseSimpleModel) {
    ONNXModel model;
    ASSERT_TRUE(model.load("../models/simple_matmul_relu.onnx"));

    ComputationGraph graph = model.parseGraph();

    // Verify at least two nodes parsed clearly (MatMul and ReLU)
    ASSERT_GE(graph.nodes.size(), 2);

    EXPECT_EQ(graph.nodes[0].op_type, "Gemm");
    EXPECT_EQ(graph.nodes[1].op_type, "Relu");

    // Verify tensors (weights) are loaded
    EXPECT_FALSE(graph.tensors.empty());
}
