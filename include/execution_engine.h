#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "graph.h"
#include "tensor.h"
#include "operators.h"

class ExecutionEngine {
public:
    ExecutionEngine();
    void executeGraph(ComputationGraph& graph, const Tensor& input);

private:
    Operators operators_;
};

#endif // EXECUTION_ENGINE_H
