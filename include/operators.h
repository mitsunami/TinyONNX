#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

class Operators {
public:
    Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, const std::vector<int>& strides, const std::vector<int>& pads, const std::vector<int>& dilations, int groups);
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor gemm(const Tensor& a, const Tensor& b, const Tensor& c, float alpha, float beta);
    Tensor gemm_transB(const Tensor& a, const Tensor& b, const Tensor& c, float alpha, float beta);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& input);
    Tensor clip(const Tensor& input, float min_val, float max_val);
    Tensor softmax(const Tensor& input);
    Tensor batchNorm(const Tensor& input, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& var, float epsilon);
    Tensor globalAveragePool(const Tensor& input);
    Tensor reshape(const Tensor& input, const std::vector<int>& new_shape);
    Tensor flatten(const Tensor& input);
};

#endif // OPERATORS_H
