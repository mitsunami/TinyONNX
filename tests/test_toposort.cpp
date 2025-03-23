#include <gtest/gtest.h>
#include "graph_utils.h"
#include "graph.h"
#include <set>

TEST(GraphUtilsTest, TopologicalSortSimpleGraph) {
    ComputationGraph graph;

    GraphNode nodeA;
    nodeA.op_type = "Constant";
    nodeA.inputs = {};
    nodeA.outputs = {"A"};

    GraphNode nodeB;
    nodeB.op_type = "Constant";
    nodeB.inputs = {};
    nodeB.outputs = {"B"};

    GraphNode nodeC;
    nodeC.op_type = "Add";
    nodeC.inputs = {"A", "B"};
    nodeC.outputs = {"C"};

    GraphNode nodeD;
    nodeD.op_type = "Relu";
    nodeD.inputs = {"C"};
    nodeD.outputs = {"D"};

    GraphNode nodeE;
    nodeE.op_type = "Mul";
    nodeE.inputs = {"C", "D"};
    nodeE.outputs = {"E"};

    // Intentionally shuffled
    graph.nodes = {nodeE, nodeD, nodeC, nodeB, nodeA};

    auto sorted = topologicalSort(graph);

    ASSERT_EQ(sorted.size(), 5);

    std::vector<std::string> output_names;
    for (const GraphNode* node : sorted) {
        ASSERT_FALSE(node->outputs.empty());
        output_names.push_back(node->outputs[0]);
    }

    // Ensure the outputs are in correct dependency order
    EXPECT_LT(std::find(output_names.begin(), output_names.end(), "A"),
              std::find(output_names.begin(), output_names.end(), "C"));
    EXPECT_LT(std::find(output_names.begin(), output_names.end(), "B"),
              std::find(output_names.begin(), output_names.end(), "C"));
    EXPECT_LT(std::find(output_names.begin(), output_names.end(), "C"),
              std::find(output_names.begin(), output_names.end(), "D"));
    EXPECT_LT(std::find(output_names.begin(), output_names.end(), "C"),
              std::find(output_names.begin(), output_names.end(), "E"));
    EXPECT_LT(std::find(output_names.begin(), output_names.end(), "D"),
              std::find(output_names.begin(), output_names.end(), "E"));
}
