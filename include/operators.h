#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

class ReLU {
public:
    void compute(Tensor& input);
};

class MatMul {
public:
    Tensor compute(const Tensor& a, const Tensor& b);
};

#endif // OPERATORS_H
