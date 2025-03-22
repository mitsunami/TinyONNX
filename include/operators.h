#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

class Operators {
public:
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& input);
};

#endif // OPERATORS_H
