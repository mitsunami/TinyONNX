#pragma once
#include <string>
#include "onnx.pb.h"
#include "graph.h"

class ONNXModel {
public:
    ONNXModel();
    bool load(const std::string& model_path);
    ComputationGraph parseGraph();

private:
    onnx::ModelProto model_proto_;
};
