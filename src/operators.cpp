#include "operators.h"
#include "utils/logger.h"
#include <xnnpack.h>
#include <pthreadpool.h>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <sstream>
#include "conv2d.cpp"

Tensor Operators::transpose(const Tensor& input, const std::vector<int>& perm) {
    std::ostringstream shape_log;
    shape_log << "TRANSPOSE: input: [" << input.shape().size() << "](" << input.shape()[0] << ", " << input.shape()[1] << ", " << input.shape()[2] << ", " << input.shape()[3] << ")";
    std::vector<int> old_shape = input.shape();
    if (perm.size() != old_shape.size())
        throw std::runtime_error("Permutation size mismatch.");

    std::vector<int> new_shape(old_shape.size());
    for (size_t i = 0; i < perm.size(); ++i)
        new_shape[i] = old_shape[perm[i]];

    Tensor output(new_shape);
    const std::vector<float>& in_data = input.data();
    std::vector<float>& out_data = output.data();

    // Compute input strides
    std::vector<int> old_strides(old_shape.size(), 1);
    for (int i = old_shape.size() - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }

    // Compute output strides
    std::vector<int> new_strides(new_shape.size(), 1);
    for (int i = new_shape.size() - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // Transpose data correctly
    for (size_t idx = 0; idx < in_data.size(); ++idx) {
        int old_idx = idx;
        std::vector<int> old_pos(old_shape.size());

        // Find the coordinate in old shape
        for (size_t i = 0; i < old_shape.size(); ++i) {
            old_pos[i] = old_idx / old_strides[i];
            old_idx %= old_strides[i];
        }

        // Permute the coordinate
        std::vector<int> new_pos(new_shape.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            new_pos[i] = old_pos[perm[i]];
        }

        // Compute new index
        int new_idx = 0;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            new_idx += new_pos[i] * new_strides[i];
        }

        out_data[new_idx] = in_data[idx];
    }
    shape_log << "         :output: [" << output.shape().size() << "](" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << ", " << output.shape()[3] << ")";
    Logger::instance().debug(shape_log.str());
    return output;
}

Tensor Operators::conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, 
                         const std::vector<int>& kernel_shape, const std::vector<int>& strides, const std::vector<int>& pads, const std::vector<int>& dilations, int groups, pthreadpool_t threadpool) {
    assert(input.shape().size() == 4);   // [N, H, W, C]
    assert(weights.shape().size() == 4); // [M, kH, kW, C/groups]
    assert(bias.shape().size() == 1);    // [M]
    std::ostringstream shape_log;
    shape_log << "CONV2D: input: [" << input.shape().size() << "](" << input.shape()[0] << ", " << input.shape()[1] << ", " << input.shape()[2] << ", " << input.shape()[3] << ")";

    const int N = input.shape()[0];
    const int IH = input.shape()[1];
    const int IW = input.shape()[2];
    const int IC = input.shape()[3];

    const int OC = weights.shape()[0];
    const int KH = weights.shape()[1];
    const int KW = weights.shape()[2];

    const int OH = (IH + 2 * pads[0] - KH) / strides[0] + 1;
    const int OW = (IW + 2 * pads[1] - KW) / strides[1] + 1;

    Tensor output({N, OH, OW, OC});

    xnn_operator_t conv_op = nullptr;
    xnn_status status = xnn_create_convolution2d_nhwc_f32(
        pads[0], pads[1], pads[2], pads[3], // top, right, bottom, left
        kernel_shape[0], kernel_shape[1],
        strides[0], strides[1],
        dilations[0], dilations[1],
        groups,
        IC / groups, OC / groups,
        IC, // input_channel_stride
        OC, // output_channel_stride
        weights.data().data(),
        bias.data().data(),
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        0,
        nullptr, // code_cache
        nullptr, // weights_cache
        &conv_op
    );
    if (status != xnn_status_success) {
        std::cout << status << std::endl;
        throw std::runtime_error("Failed to create XNNPACK convolution operator");
    }

    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    status = xnn_reshape_convolution2d_nhwc_f32(
        conv_op,
        1, //batch_size
        IH, IW,
        &workspace_size, &workspace_alignment,
        nullptr, // output_height_out
        nullptr, // output_width_out
        threadpool
    );
    if (status != xnn_status_success) {
        throw std::runtime_error("Failed to reshape XNNPACK convolution operator");
    }

    std::vector<char> workspace(workspace_size);
    status = xnn_setup_convolution2d_nhwc_f32(
        conv_op,
        workspace.data(),
        input.data().data(),
        output.data().data()
    );
    if (status != xnn_status_success) {
        std::cout << status << std::endl;
        throw std::runtime_error("Failed to set up XNNPACK convolution operator");
    }

    status = xnn_run_operator(conv_op, threadpool);
    if (status != xnn_status_success) {
        throw std::runtime_error("Failed to run XNNPACK convolution operator");
    }

    status = xnn_delete_operator(conv_op);
    if (status != xnn_status_success) {
        throw std::runtime_error("Failed to delete XNNPACK convolution operator");
    }

    conv_op = nullptr;

    shape_log << "      :output: [" << output.shape().size() << "](" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << ", " << output.shape()[3] << ")";
    Logger::instance().debug(shape_log.str());

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
    std::ostringstream shape_log;
    shape_log << "GEMM_TRANSB A: (" << a.shape()[0] << ", " << a.shape()[1] << "), B: (" << b.shape()[0] << ", " << b.shape()[1] <<")";
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
    Logger::instance().debug(shape_log.str());
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
    assert(input.shape().size() == 4); // [batch, height, width, channels]
    std::ostringstream shape_log;
    shape_log << "GLOBALAVGPOOL: input: [" << input.shape().size() << "](" << input.shape()[0] << ", " << input.shape()[1] << ", " << input.shape()[2] << ", " << input.shape()[3] << ")";

    int batch = input.shape()[0];
    int height = input.shape()[1];
    int width = input.shape()[2];
    int channels = input.shape()[3];

    Tensor output({batch, 1, 1, channels});
    int spatial_size = height * width;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = ((b * height + h) * width + w) * channels + c;
                    sum += input.data()[idx];
                }
            }
            int out_idx = (b * channels + c);
            output.data()[out_idx] = sum / spatial_size;
        }
    }

    shape_log << "      :output: [" << output.shape().size() << "](" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << ", " << output.shape()[3] << ")";
    Logger::instance().debug(shape_log.str());
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

Tensor Operators::flatten(const Tensor& input, int axis) {
    // TODO: support axis
    assert(input.shape().size() >= 2);
    std::ostringstream shape_log;
    shape_log << "FLATTEN: input: [" << input.shape().size() << "](" << input.shape()[0] << ", " << input.shape()[1] << ", " << input.shape()[2] << ", " << input.shape()[3] << ")";
    int batch = input.shape()[0];
    int features = input.data().size() / batch;

    Tensor output = reshape(input, {batch, features});
    shape_log << "       : output: [" << output.shape().size() << "](" << output.shape()[0] << ", " << output.shape()[1] << ")";
    Logger::instance().debug(shape_log.str());

    return output;
}
