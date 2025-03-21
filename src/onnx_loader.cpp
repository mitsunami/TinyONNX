#include "onnx_loader.h"
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
