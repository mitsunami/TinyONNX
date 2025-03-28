#include "onnx_loader.h"
#include "graph.h"
//#include "graph_utils.h"
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
    std::string input_name = graph_proto.input(0).name();
    std::string output_name = graph_proto.output(0).name();

    // Check if model contains ops sensitive to channels-last
    const auto& input_shape_proto = graph_proto.input(0).type().tensor_type().shape();
    bool is_input_4d = (input_shape_proto.dim_size() == 4);
    bool requires_channel_last = false;
    for (const auto& node_proto : graph_proto.node()) {
        if (node_proto.op_type() == "Conv" || 
            node_proto.op_type() == "MaxPool" || 
            node_proto.op_type() == "AveragePool" || 
            node_proto.op_type() == "BatchNormalization") {
            requires_channel_last = true;
            break;
        }
    }
    bool insert_global_transpose  = (input_shape_proto.dim_size() == 4);

    // Parse initializers (constants: weights, biases)
    for (const auto& initializer : graph_proto.initializer()) {
        Tensor tensor({initializer.dims().begin(), initializer.dims().end()});
        memcpy(tensor.data().data(), initializer.raw_data().data(), initializer.raw_data().size());
        graph.tensors[initializer.name()] = tensor;
    }

    if(insert_global_transpose){
        // Insert global input transpose (NCHW -> NHWC)
        GraphNode preTranspose;
        preTranspose.op_type = "Transpose";
        preTranspose.inputs = {input_name};
        preTranspose.outputs = {input_name + "_nhwc"};
        onnx::AttributeProto perm_attr;
        perm_attr.set_name("perm");
        perm_attr.set_type(onnx::AttributeProto::INTS);
        perm_attr.add_ints(0);
        perm_attr.add_ints(2);
        perm_attr.add_ints(3);
        perm_attr.add_ints(1);
        preTranspose.attributes.push_back(perm_attr);
        graph.nodes.push_back(preTranspose);
    }

    // Parse graph nodes
    for (const auto& node_proto : graph_proto.node()) {
        GraphNode node;
        node.op_type = node_proto.op_type();
        node.inputs.assign(node_proto.input().begin(), node_proto.input().end());
        node.outputs.assign(node_proto.output().begin(), node_proto.output().end());
        node.attributes.assign(node_proto.attribute().begin(), node_proto.attribute().end());
        // for (const auto& attr : node_proto.attribute()) {
        //     node.attributes.push_back(attr);
        // }

        if(requires_channel_last){
            if(node.op_type == "Conv") {
                // Transpose weights for NHWC compatibility
                graph.tensors[node.inputs[1]].reorderOIHWtoOHWI();
                node.inputs[0] += "_nhwc";
                node.outputs[0] += "_nhwc";
            } else {
                // Update non-Conv nodes to handle NHWC if needed
                for(auto& input : node.inputs) {
                    if(input == input_name)
                        input += "_nhwc"; // explicitly update first node input
                    else if (graph.tensors.find(input + "_nhwc") != graph.tensors.end())
                        input += "_nhwc";
                }
                for (auto& output : node.outputs) {
                    if(output == output_name)
                        output += "_nhwc";
                }
            }
        }

        graph.nodes.push_back(node);
    }

    if(insert_global_transpose){
        // Insert global output transpose (NHWC -> NCHW)
        GraphNode postTranspose;
        postTranspose.op_type = "Transpose";
        postTranspose.inputs = {output_name + "_nhwc"};
        postTranspose.outputs = {output_name};
        onnx::AttributeProto perm_attr_out;
        perm_attr_out.set_name("perm");
        perm_attr_out.set_type(onnx::AttributeProto::INTS);
        perm_attr_out.add_ints(0);
        perm_attr_out.add_ints(3);
        perm_attr_out.add_ints(1);
        perm_attr_out.add_ints(2);
        postTranspose.attributes.push_back(perm_attr_out);
        graph.nodes.push_back(postTranspose);
    }

    graph.printNodes();
    graph.topologicalSort();
    graph.printSortedNodes();

    return graph;
}
