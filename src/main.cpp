#include <iostream>
#include "onnx_loader.h"
#include "execution_engine.h"
#include "tensor.h"
#include "utils/timer.h"
#include "utils/meminfo.h"
#include "utils/logger.h"
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
    std::vector<std::string> args(argv + 1, argv + argc);

    // Check for --debug flag
    bool debug_enabled = false;
    std::vector<std::string> positional_args;

    for (const auto& arg : args) {
        if (arg == "--debug") {
            debug_enabled = true;
        } else {
            positional_args.push_back(arg);
        }
    }

    // Set logging level
    Logger::instance().setLevel(debug_enabled ? LOG_LEVEL_DEBUG : LOG_LEVEL_INFO);
    Logger::instance().debug("Debugging is enabled");
    
    // Expect exactly 2 positional arguments: model and input file
    if (positional_args.size() != 2) {
        Logger::instance().error("Usage: <program> [--debug] <onnx_model> <input_tensor.npy>");
        return 1;
    }
    
    std::string model_path = positional_args[0];
    std::string input_path = positional_args[1];

    Logger::instance().info("Program started");

    ONNXModel model;
    if (!model.load(model_path)) {
        Logger::instance().error("Error: Unable to load ONNX model.");
        return 1;
    }

    ComputationGraph graph = model.parseGraph();
    Tensor input = loadNumpyInput(input_path);

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
