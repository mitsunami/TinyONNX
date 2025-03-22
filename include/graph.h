#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <unordered_map>
#include "tensor.h"

struct GraphNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

struct ComputationGraph {
    std::vector<GraphNode> nodes;
    std::unordered_map<std::string, Tensor> tensors;
};

#endif // GRAPH_H
