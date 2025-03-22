#ifndef ONNX_LOADER_H
#define ONNX_LOADER_H

#include <string>
#include <onnx/onnx_pb.h>
#include "graph.h"

class ONNXModel {
public:
    ONNXModel();
    bool load(const std::string& model_path);
    ComputationGraph parseGraph();

private:
    onnx::ModelProto model_proto_;
};

#endif // ONNX_LOADER_H
