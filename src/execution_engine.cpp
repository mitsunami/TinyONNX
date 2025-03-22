#include "execution_engine.h"
#include "operators.h"
#include <iostream>

ExecutionEngine::ExecutionEngine() {}

void ExecutionEngine::executeGraph(ComputationGraph& graph, const Tensor& input) {
    std::cout << "Running inference..." << std::endl;

    graph.tensors["input"] = input;

    for (const auto& node : graph.nodes) {
        std::cout << "Executing Node: " << node.op_type << std::endl;

        if (node.op_type == "MatMul") {
            Tensor& a = graph.tensors[node.inputs[0]];
            Tensor& b = graph.tensors[node.inputs[1]];
            Tensor output = operators_.matmul(a, b);
            graph.tensors[node.outputs[0]] = output;
        }
        else if (node.op_type == "Relu") {
            Tensor& input_tensor = graph.tensors[node.inputs[0]];
            Tensor output = operators_.relu(input_tensor);
            graph.tensors[node.outputs[0]] = output;
        }
        else {
            std::cerr << "Operator not supported yet: " << node.op_type << std::endl;
        }
    }
    
    // Show final output tensor (assuming named 'output')
    if (graph.tensors.count("output")) {
        std::cout << "Final Output Tensor: ";
        graph.tensors["output"].print();
    } else {
        std::cerr << "Output tensor not found!" << std::endl;
    }

}
