#include "operators.h"
#include <iostream>
#include <cassert>

void ReLU::compute(Tensor& input) {
    input.applyReLU();
}

Tensor MatMul::compute(const Tensor& a, const Tensor& b) {
    assert(a.shape().size() == 2 && b.shape().size() == 2);
    int m = a.shape()[0];
    int k = a.shape()[1];
    int n = b.shape()[1];
    assert(k == b.shape()[0]);

    Tensor output({m, n});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a.data()[i * k + l] * b.data()[l * n + j];
            }
            output.data()[i * n + j] = sum;
        }
    }
    return output;
}
