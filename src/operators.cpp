#include "operators.h"
#include <cmath>
#include <cassert>
#include <algorithm>

Tensor Operators::conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, const std::vector<int>& strides, const std::vector<int>& pads, const std::vector<int>& dilations, int groups) {
    assert(input.shape().size() == 4);   // [N, C, H, W]
    assert(weights.shape().size() == 4); // [M, C/groups, kH, kW]
    assert(bias.shape().size() == 1);    // [M]

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];

    int M = weights.shape()[0];          // Output channels
    int kC = weights.shape()[1];         // Input channels per group
    int kH = weights.shape()[2];
    int kW = weights.shape()[3];

    int stride_h = strides[0];
    int stride_w = strides[1];

    int pad_top = pads[0];
    int pad_left = pads[1];
    int pad_bottom = pads[2];
    int pad_right = pads[3];

    int dilation_h = dilations[0];
    int dilation_w = dilations[1];

    int out_h = (H + pad_top + pad_bottom - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_w = (W + pad_left + pad_right - dilation_w * (kW - 1) - 1) / stride_w + 1;

    Tensor output({N, M, out_h, out_w});

    // Pad input (zero padding)
    Tensor padded_input({N, C, H + pad_top + pad_bottom, W + pad_left + pad_right});
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    int pi = ((n * C + c) * (H + pad_top + pad_bottom) + (h + pad_top)) * (W + pad_left + pad_right) + (w + pad_left);
                    int ii = ((n * C + c) * H + h) * W + w;
                    padded_input.data()[pi] = input.data()[ii];
                }

    // Convolution loop
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            int g = m / (M / groups); // group index
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias.data()[m];

                    for (int c = 0; c < kC; ++c) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride_h + kh * dilation_h;
                                int iw = ow * stride_w + kw * dilation_w;
                                int in_c = g * kC + c;

                                int pi = ((n * C + in_c) * (H + pad_top + pad_bottom) + ih) * (W + pad_left + pad_right) + iw;
                                int wi = ((m * kC + c) * kH + kh) * kW + kw;

                                sum += padded_input.data()[pi] * weights.data()[wi];
                            }
                        }
                    }

                    int oi = ((n * M + m) * out_h + oh) * out_w + ow;
                    output.data()[oi] = sum;
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

Tensor Operators::gemm(const Tensor& a, const Tensor& b, const Tensor& c, float alpha, float beta) {
    assert(a.shape().size() == 2 && b.shape().size() == 2);
    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];
    assert(K == b.shape()[0]);
    assert(c.data().size() == N);

    Tensor result({M, N});

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = c.data()[n]; // Add bias initially
            for (int k = 0; k < K; ++k) {
                sum += a.data()[m * K + k] * b.data()[k * N + n];
            }
            result.data()[m * N + n] = sum;
        }
    }
    return result;
}

Tensor Operators::gemm_transB(const Tensor& a, const Tensor& b, const Tensor& c, float alpha, float beta) {
    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[0];  // B shape is [N, K], so B^T is [K, N]

    assert(K == b.shape()[1]);
    assert(c.data().size() == N);

    Tensor result({M, N});

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a.data()[m * K + k] * b.data()[n * K + k];  // B^T
            }
            result.data()[m * N + n] = alpha * sum + beta * c.data()[n];
        }
    }
    return result;
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

Tensor Operators::clip(const Tensor& input, float min_val, float max_val) {
    Tensor output(input.shape());
    for (size_t i = 0; i < input.data().size(); ++i)
        output.data()[i] = std::max(min_val, std::min(max_val, input.data()[i]));
    return output;
}

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

Tensor Operators::flatten(const Tensor& input) {
    assert(input.shape().size() >= 2);
    int batch = input.shape()[0];
    int features = input.data().size() / batch;

    return reshape(input, {batch, features});
}
