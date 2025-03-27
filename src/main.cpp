#include <iostream>
#include "onnx_loader.h"
#include "execution_engine.h"
#include "tensor.h"
#include "utils/timer.h"
#include "utils/meminfo.h"
#include <fstream>

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

    Timer total_timer("Total Graph Execution");
    ExecutionEngine engine;
    engine.executeGraph(graph, input);

    #ifdef ENABLE_MEM_USAGE
    printPeakRSS();
    #endif

    // Show final output tensor (assuming named 'output')
    if (graph.tensors.count("output")) {
        graph.tensors["output"].print();
        std::ofstream fout("tinyonnx_output.txt");
        for (float val : graph.tensors["output"].data()) {
            fout << val << "\n";
        }
        fout.close();
    } else {
        std::cerr << "Output tensor not found!" << std::endl;
    }

    std::cout << "ONNX Model execution completed!" << std::endl;
    return 0;
}
