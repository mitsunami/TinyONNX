#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

class Operators {
public:
    Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, 
                  int stride, int padding);
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& input);
};

#endif // OPERATORS_H
