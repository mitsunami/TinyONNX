#include "execution_engine.h"
#include "operators.h"
#include <onnx/onnx_pb.h>
#include <iostream>

ExecutionEngine::ExecutionEngine() {}

void ExecutionEngine::executeGraph(ComputationGraph& graph, const Tensor& input) {
    std::cout << "Running inference..." << std::endl;

    graph.tensors["input"] = input;

    for (const auto& node : graph.nodes) {
        std::cout << "Executing Node: " << node.op_type << std::endl;

        if (node.op_type == "Constant") {
            assert(!node.attributes.empty());
            const onnx::AttributeProto& attr = node.attributes[0];
            assert(attr.has_t());
            const onnx::TensorProto& tensor_proto = attr.t();
            Tensor tensor({tensor_proto.dims().begin(), tensor_proto.dims().end()});
            memcpy(tensor.data().data(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
            graph.tensors[node.outputs[0]] = tensor;
        }
        else if (node.op_type == "Conv") {
            auto& in = graph.tensors[node.inputs[0]];
            auto& weights = graph.tensors[node.inputs[1]];
            auto& bias = graph.tensors[node.inputs[2]];
            int stride = 1;    // TODO: Retrieve from node attributes later clearly
            int padding = 1;   // TODO: Retrieve from node attributes later clearly
            graph.tensors[node.outputs[0]] = operators_.conv2d(in, weights, bias, stride, padding);
        }
        else if (node.op_type == "MatMul") {
            auto& a = graph.tensors[node.inputs[0]];
            auto& b = graph.tensors[node.inputs[1]];
            graph.tensors[node.outputs[0]] = operators_.matmul(a, b);
        }
        else if (node.op_type == "Gemm") {
            auto& A = graph.tensors[node.inputs[0]];
            auto& B = graph.tensors[node.inputs[1]];
            auto& C = graph.tensors[node.inputs[2]]; // Bias
            Tensor result = operators_.matmul(A, B);        
            // Add bias
            for (size_t i = 0; i < result.data().size(); ++i)
                result.data()[i] += C.data()[i];
            graph.tensors[node.outputs[0]] = result;
        }
        else if (node.op_type == "Add") {
            auto& a = graph.tensors[node.inputs[0]];
            auto& b = graph.tensors[node.inputs[1]];
            graph.tensors[node.outputs[0]] = operators_.add(a, b);
        }
        else if (node.op_type == "Relu") {
            auto& input_tensor = graph.tensors[node.inputs[0]];
            graph.tensors[node.outputs[0]] = operators_.relu(input_tensor);
        }
        else if (node.op_type == "Softmax") {
            auto& input_tensor = graph.tensors[node.inputs[0]];
            graph.tensors[node.outputs[0]] = operators_.softmax(input_tensor);
        }
        else if (node.op_type == "BatchNormalization") {
            auto& in = graph.tensors[node.inputs[0]];
            auto& scale = graph.tensors[node.inputs[1]];
            auto& bias = graph.tensors[node.inputs[2]];
            auto& mean = graph.tensors[node.inputs[3]];
            auto& var = graph.tensors[node.inputs[4]];
            float epsilon = 1e-5f; // TODO: later parse from attributes clearly
            graph.tensors[node.outputs[0]] = operators_.batchNorm(in, scale, bias, mean, var, epsilon);
        }
        else if (node.op_type == "GlobalAveragePool") {
            auto& in = graph.tensors[node.inputs[0]];
            graph.tensors[node.outputs[0]] = operators_.globalAveragePool(in);
        }
        else if (node.op_type == "Reshape") {
            auto& in = graph.tensors[node.inputs[0]];
            auto& shape_tensor = graph.tensors[node.inputs[1]];
            std::vector<int> new_shape(shape_tensor.data().begin(), shape_tensor.data().end());
            graph.tensors[node.outputs[0]] = operators_.reshape(in, new_shape);
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
