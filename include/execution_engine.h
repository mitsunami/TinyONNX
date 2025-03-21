#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "onnx_loader.h"
#include "tensor.h"

class ExecutionEngine {
public:
    ExecutionEngine();
    void run(const ONNXModel& model);
};

#endif // EXECUTION_ENGINE_H
