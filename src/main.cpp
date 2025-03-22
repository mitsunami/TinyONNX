#include <iostream>
#include "onnx_loader.h"
#include "execution_engine.h"
#include "tensor.h"
#include <fstream>

/*
>>> print("Top 5 predictions (indices):", top5.indices.numpy())
Top 5 predictions (indices): [[644 904 446 549 971]]
>>> print("Top 5 predictions (scores):", top5.values.detach().numpy())
Top 5 predictions (scores): [[6.4933834 6.2849607 6.1947975 6.137249  5.4067903]]
*/

Tensor loadNumpyInput(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open input_tensor.npy");
    }
    file.seekg(128); // Skip numpy header (assume standard header size for simplicity)
    std::vector<float> data(1 * 3 * 224 * 224);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return Tensor({1, 3, 224, 224}, data);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model> <input_tensor.npy>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    ONNXModel model;
    if (!model.load(model_path)) {
        std::cerr << "Failed to load ONNX model!" << std::endl;
        return 1;
    }

    ComputationGraph graph = model.parseGraph();

    Tensor input = loadNumpyInput(argv[2]);

    ExecutionEngine engine;
    engine.executeGraph(graph, input);

    graph.tensors["output"].print();

    std::cout << "ONNX Model execution completed!" << std::endl;
    return 0;
}
