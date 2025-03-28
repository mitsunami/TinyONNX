#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "onnx.pb.h"
#include "tensor.h"

struct GraphNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<onnx::AttributeProto> attributes; 
};

class ComputationGraph {
public:
    std::vector<GraphNode> nodes; // original order
    std::vector<const GraphNode*> sorted_nodes; // topologically sorted
    std::unordered_map<std::string, Tensor> tensors;

    void topologicalSort();
    void printNodes();
    void printSortedNodes();
};
