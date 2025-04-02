#include "execution_engine.h"
#include <xnnpack.h>
#include "operators.h"
#include "onnx_utils.h"
#include "graph.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "onnx.pb.h"
#include <iostream>

ExecutionEngine::ExecutionEngine() : pthreadpool_(nullptr) {
    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
        throw std::runtime_error("XNNPACK initialization failed");
    }
    pthreadpool_ = pthreadpool_create(0); // Use all hardware threads
}

ExecutionEngine::~ExecutionEngine() {
    xnn_deinitialize();
    if (pthreadpool_) pthreadpool_destroy(pthreadpool_);
}

void ExecutionEngine::executeGraph(ComputationGraph& graph, const Tensor& input) {
    graph.tensors["input"] = input;

    for (const GraphNode* node : graph.sorted_nodes) {
        Timer timer("Op: " + node->op_type);

        if (node->op_type == "Constant") {
            assert(!node->attributes.empty());
            const onnx::AttributeProto& attr = node->attributes[0];
            assert(attr.has_t());
            const onnx::TensorProto& tensor_proto = attr.t();
            Tensor tensor({tensor_proto.dims().begin(), tensor_proto.dims().end()});
            memcpy(tensor.data().data(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
            graph.tensors[node->outputs[0]] = tensor;
        }
        else if (node->op_type == "Conv") {
            auto& in = graph.tensors[node->inputs[0]];
            auto& weights = graph.tensors[node->inputs[1]];
            auto& bias = graph.tensors[node->inputs[2]];
            std::vector<int> kernel_shape = getIntListAttr(node, "kernel_shape");
            std::vector<int> strides = getIntListAttr(node, "strides");
            std::vector<int> pads = getIntListAttr(node, "pads");
            std::vector<int> dilations = getIntListAttr(node, "dilations");
            int groups = getIntAttr(node, "group", 1);        
            if (strides.empty()) strides = {1, 1};
            if (pads.empty()) pads = {0, 0, 0, 0};  // top, left, bottom, right
            if (dilations.empty()) dilations = {1, 1};
            graph.tensors[node->outputs[0]] = operators_.conv2d(
                in, weights, bias, kernel_shape, strides, pads, dilations, groups, pthreadpool_
            );
        }
        else if (node->op_type == "Transpose") {
            auto& in = graph.tensors[node->inputs[0]];
            auto perm = getIntListAttr(node, "perm");
            graph.tensors[node->outputs[0]] = operators_.transpose(in, perm);
        }
        else if (node->op_type == "MatMul") {
            auto& a = graph.tensors[node->inputs[0]];
            auto& b = graph.tensors[node->inputs[1]];
            graph.tensors[node->outputs[0]] = operators_.matmul(a, b);
        }
        else if (node->op_type == "Gemm") {
            auto& in = graph.tensors[node->inputs[0]];
            auto& weights = graph.tensors[node->inputs[1]];
            auto& bias = graph.tensors[node->inputs[2]];
            float alpha = getFloatAttr(node, "alpha", 1.0f);
            float beta = getFloatAttr(node, "beta", 1.0f);
            bool transB = getIntAttr(node, "transB", 0);
            Tensor result;
            if (transB) {
                result = operators_.gemm_transB(in, weights, bias, alpha, beta);
            } else {
                result = operators_.gemm(in, weights, bias, alpha, beta);
            }
            graph.tensors[node->outputs[0]] = result;
        }
        else if (node->op_type == "Add") {
            auto& a = graph.tensors[node->inputs[0]];
            auto& b = graph.tensors[node->inputs[1]];
            graph.tensors[node->outputs[0]] = operators_.add(a, b);
        }
        else if (node->op_type == "Relu") {
            auto& input_tensor = graph.tensors[node->inputs[0]];
            graph.tensors[node->outputs[0]] = operators_.relu(input_tensor);
        }
        else if (node->op_type == "Clip") {
            auto& in = graph.tensors[node->inputs[0]];
            float min_val = getFloatAttr(node, "min", 0.0f);
            float max_val = getFloatAttr(node, "max", 6.0f);
            graph.tensors[node->outputs[0]] = operators_.clip(in, min_val, max_val);
        }
        else if (node->op_type == "Softmax") {
            auto& input_tensor = graph.tensors[node->inputs[0]];
            graph.tensors[node->outputs[0]] = operators_.softmax(input_tensor);
        }
        else if (node->op_type == "BatchNormalization") {
            auto& in = graph.tensors[node->inputs[0]];
            auto& scale = graph.tensors[node->inputs[1]];
            auto& bias = graph.tensors[node->inputs[2]];
            auto& mean = graph.tensors[node->inputs[3]];
            auto& var = graph.tensors[node->inputs[4]];
            float epsilon = 1e-5f; // TODO: later parse from attributes clearly
            graph.tensors[node->outputs[0]] = operators_.batchNorm(in, scale, bias, mean, var, epsilon);
        }
        else if (node->op_type == "GlobalAveragePool") {
            auto& in = graph.tensors[node->inputs[0]];
            graph.tensors[node->outputs[0]] = operators_.globalAveragePool(in);
        }
        else if (node->op_type == "MaxPool") {
            auto& in = graph.tensors[node->inputs[0]];
            int ceil_mode = getIntAttr(node, "ceil_mode", 0);
            std::vector<int> dilations = getIntListAttr(node, "dilations");
            std::vector<int> kernel_shape = getIntListAttr(node, "kernel_shape");
            std::vector<int> pads = getIntListAttr(node, "pads");
            std::vector<int> strides = getIntListAttr(node, "strides");
            if (strides.empty()) strides = {1, 1};
            if (pads.empty()) pads = {0, 0, 0, 0};  // top, left, bottom, right
            if (dilations.empty()) dilations = {1, 1};
            graph.tensors[node->outputs[0]] = operators_.maxPool(
                in, ceil_mode, dilations, kernel_shape, pads, strides, pthreadpool_
            );
        }
        else if (node->op_type == "Reshape") {
            auto& in = graph.tensors[node->inputs[0]];
            auto& shape_tensor = graph.tensors[node->inputs[1]];
            std::vector<int> new_shape(shape_tensor.data().begin(), shape_tensor.data().end());
            graph.tensors[node->outputs[0]] = operators_.reshape(in, new_shape);
        }
        else if (node->op_type == "Flatten") {
            auto& in = graph.tensors[node->inputs[0]];
            int axis = getIntAttr(node, "axis", 0);
            graph.tensors[node->outputs[0]] = operators_.flatten(in, axis);
        }
        else {
            std::cerr << "Operator not supported yet: " << node->op_type << std::endl;
        }
    }
    
}
