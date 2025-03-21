#include "execution_engine.h"
#include "operators.h"
#include <iostream>

ExecutionEngine::ExecutionEngine() {}

void ExecutionEngine::run(const ONNXModel& model) {
    std::cout << "Running inference..." << std::endl;

    Tensor input({1, 3, 224, 224});
    input.fillRandom();
    input.print();

    Tensor weights({224, 1000});
    weights.fillRandom();

    MatMul matmul;
    Tensor output = matmul.compute(input, weights);
    output.print();

    ReLU relu;
    relu.compute(output);
    output.print();

    // TODO: Complete execution graph logic
}
