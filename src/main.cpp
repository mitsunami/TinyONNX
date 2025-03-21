#include <iostream>
#include "onnx_loader.h"
#include "execution_engine.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    ONNXModel model;
    if (!model.load(model_path)) {
        std::cerr << "Failed to load ONNX model!" << std::endl;
        return 1;
    }

    ExecutionEngine engine;
    engine.run(model);

    std::cout << "ONNX Model execution completed!" << std::endl;
    return 0;
}
