#include "onnx_loader.h"
#include "graph.h"
#include <iostream>
#include <fstream>

ONNXModel::ONNXModel() {}

bool ONNXModel::load(const std::string& model_path) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        std::cerr << "Error: Unable to open model file." << std::endl;
        return false;
    }

    if (!model_proto_.ParseFromIstream(&input)) {
        std::cerr << "Error: Failed to parse ONNX model." << std::endl;
        return false;
    }

    std::cout << "ONNX Model successfully loaded." << std::endl;
    return true;
}

ComputationGraph ONNXModel::parseGraph() {
    ComputationGraph graph;

    const auto& graph_proto = model_proto_.graph();

    // Parse initializers (constants: weights, biases)
    for (const auto& initializer : graph_proto.initializer()) {
        Tensor tensor({initializer.dims().begin(), initializer.dims().end()});
        memcpy(tensor.data().data(), initializer.raw_data().data(), initializer.raw_data().size());
        graph.tensors[initializer.name()] = tensor;
    }

    // Parse graph nodes
    for (const auto& node_proto : graph_proto.node()) {
        GraphNode node;
        node.op_type = node_proto.op_type();
        node.inputs.assign(node_proto.input().begin(), node_proto.input().end());
        node.outputs.assign(node_proto.output().begin(), node_proto.output().end());
        for (const auto& attr : node_proto.attribute()) {
            node.attributes.push_back(attr);
        }
        graph.nodes.push_back(node);
    }

    return graph;
}
