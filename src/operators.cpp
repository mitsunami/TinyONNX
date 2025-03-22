#include "operators.h"
#include <cmath>
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

Tensor Operators::add(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape()); // Ensure tensors have identical shapes

    Tensor output(a.shape());

    size_t total_elements = a.data().size();
    for (size_t i = 0; i < total_elements; ++i) {
        output.data()[i] = a.data()[i] + b.data()[i];
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

#include <cmath>

Tensor Operators::softmax(const Tensor& input) {
    Tensor output(input.shape());
    size_t total_elements = input.data().size();

    // Find max for numerical stability
    float max_val = *std::max_element(input.data().begin(), input.data().end());

    float sum_exp = 0.0f;
    for (size_t i = 0; i < total_elements; ++i) {
        output.data()[i] = std::exp(input.data()[i] - max_val);
        sum_exp += output.data()[i];
    }

    for (size_t i = 0; i < total_elements; ++i) {
        output.data()[i] /= sum_exp;
    }

    return output;
}

Tensor Operators::batchNorm(const Tensor& input, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& var, float epsilon) {
    assert(input.shape().size() == 4);  // [batch, channels, height, width]

    Tensor output(input.shape());

    int batch = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float s = scale.data()[c];
            float b_ = bias.data()[c];
            float m = mean.data()[c];
            float v = var.data()[c];

            float denom = 1.0f / sqrt(v + epsilon);

            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    output.data()[idx] = s * (input.data()[idx] - m) * denom + b_;
                }
            }
        }
    }

    return output;
}

Tensor Operators::globalAveragePool(const Tensor& input) {
    assert(input.shape().size() == 4); // [batch, channels, height, width]

    int batch = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    Tensor output({batch, channels, 1, 1});

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    sum += input.data()[idx];
                }
            }
            float avg = sum / (height * width);
            int out_idx = (b * channels + c);
            output.data()[out_idx] = avg;
        }
    }

    return output;
}

Tensor Operators::reshape(const Tensor& input, const std::vector<int>& new_shape) {
    size_t input_size = input.data().size();

    // Calculate new shape size
    size_t new_size = 1;
    int infer_dim = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            infer_dim = i;
        } else {
            new_size *= new_shape[i];
        }
    }

    std::vector<int> final_shape = new_shape;
    if (infer_dim != -1) {
        final_shape[infer_dim] = input_size / new_size;
        new_size *= final_shape[infer_dim];
    }

    assert(input_size == new_size);

    Tensor output(final_shape);
    output.data() = input.data(); // Just copy data, no change

    return output;
}
