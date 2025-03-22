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
            auto& in = graph.tensors[node.inputs[0]];
            auto& weights = graph.tensors[node.inputs[1]];
            auto& bias = graph.tensors[node.inputs[2]];
            float alpha = 1.0f;
            float beta = 1.0f;
            bool transB = false;
            for (const auto& attr : node.attributes) {
                if (attr.name() == "alpha") alpha = attr.f();
                if (attr.name() == "beta") beta = attr.f();
                if (attr.name() == "transB") transB = (attr.i() != 0);
            }
            Tensor result;
            if (transB) {
                result = operators_.gemm_transB(in, weights, bias, alpha, beta);
            } else {
                result = operators_.gemm(in, weights, bias, alpha, beta);
            }
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
        else if (node.op_type == "Clip") {
            auto& in = graph.tensors[node.inputs[0]];
            float min_val = 0.0f; // default
            float max_val = 6.0f; // default
            for (const auto& attr : node.attributes) {
                if (attr.name() == "min") min_val = attr.f();
                if (attr.name() == "max") max_val = attr.f();
            }
            graph.tensors[node.outputs[0]] = operators_.clip(in, min_val, max_val);
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
        else if (node.op_type == "Flatten") {
            auto& in = graph.tensors[node.inputs[0]];
            graph.tensors[node.outputs[0]] = operators_.flatten(in);
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
