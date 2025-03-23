#pragma once
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
