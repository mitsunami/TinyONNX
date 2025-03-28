#pragma once
#include "graph.h"
#include "tensor.h"
#include "operators.h"
#include <pthreadpool.h>

class ExecutionEngine {
public:
    ExecutionEngine();
    ~ExecutionEngine();
    void executeGraph(ComputationGraph& graph, const Tensor& input);

private:
    pthreadpool_t pthreadpool_;
    Operators operators_;
};
