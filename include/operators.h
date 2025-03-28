#pragma once
#include "tensor.h"
#include <pthreadpool.h>

class Operators {
public:
    Tensor transpose(const Tensor& input, const std::vector<int>& perm);
    Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, const std::vector<int>& kernel_shape, const std::vector<int>& strides, const std::vector<int>& pads, const std::vector<int>& dilations, int groups, pthreadpool_t threadpool);
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
    Tensor flatten(const Tensor& input, int axis);
};
