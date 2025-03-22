#include "operators.h"
#include <cassert>
#include <algorithm>

Tensor Operators::conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, int stride, int padding) {
    assert(input.shape().size() == 4);   // [batch, in_channels, height, width]
    assert(weights.shape().size() == 4); // [out_channels, in_channels, kernel_h, kernel_w]
    assert(bias.shape().size() == 1);    // [out_channels]

    int batch = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_height = input.shape()[2];
    int in_width = input.shape()[3];

    int out_channels = weights.shape()[0];
    int kernel_h = weights.shape()[2];
    int kernel_w = weights.shape()[3];

    int out_height = (in_height - kernel_h + 2 * padding) / stride + 1;
    int out_width = (in_width - kernel_w + 2 * padding) / stride + 1;

    Tensor output({batch, out_channels, out_height, out_width});

    // Naive implementation
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = bias.data()[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                    sum += input.data()[input_idx] * weights.data()[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output.data()[output_idx] = sum;
                }
            }
        }
    }

    return output;
}

Tensor Operators::matmul(const Tensor& a, const Tensor& b) {
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

Tensor Operators::relu(const Tensor& input) {
    Tensor output(input.shape());

    for (size_t i = 0; i < input.data().size(); ++i) {
        output.data()[i] = std::max(0.0f, input.data()[i]);
    }

    return output;
}
