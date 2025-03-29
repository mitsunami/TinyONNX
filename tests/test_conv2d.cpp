#include <gtest/gtest.h>
#include <xnnpack.h>
#include "operators.h"
#include "tensor.h"

TEST(Conv2DTest, BasicNoPaddingStride1) {
    Tensor input({1, 4, 4, 1});
    Tensor weights({1, 3, 3, 1});
    Tensor bias({1});

    input.data() = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16
    };
    weights.data() = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    bias.data()[0] = 0.0f;

    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
        throw std::runtime_error("XNNPACK initialization failed");
    }
    pthreadpool_t pthreadpool_ = pthreadpool_create(0);
    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {3, 3},      // kernel_size
        {1, 1},      // stride
        {0, 0, 0, 0},// padding (top, left, bottom, right)
        {1, 1},      // dilation
        1,            // groups
        pthreadpool_
    );

    std::vector<float> expected = {-6, -6, -6, -6};

    ASSERT_EQ(output.shape(), std::vector<int>({1, 2, 2, 1}));
    EXPECT_FLOAT_EQ(output.data()[0], -6);  // top-left
    EXPECT_FLOAT_EQ(output.data()[1], -6);  // top-right
    EXPECT_FLOAT_EQ(output.data()[2], -6);  // bottom-left
    EXPECT_FLOAT_EQ(output.data()[3], -6);  // bottom-right

    if (pthreadpool_) pthreadpool_destroy(pthreadpool_);
}

TEST(Conv2DTest, Stride2Padding1) {
    Tensor input({1, 4, 4, 1});
    Tensor weights({1, 3, 3, 1});
    Tensor bias({1});

    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
        throw std::runtime_error("XNNPACK initialization failed");
    }
    pthreadpool_t pthreadpool_ = pthreadpool_create(0);
    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {3, 3},      // kernel_size
        {2, 2},
        {1, 1, 1, 1},
        {1, 1},
        1,
        pthreadpool_
    );

    EXPECT_EQ(output.shape(), std::vector<int>({1, 2, 2, 1}));
    if (pthreadpool_) pthreadpool_destroy(pthreadpool_);
}

TEST(Conv2DTest, DepthwiseGroupConv) {
    Tensor input({1, 4, 4, 2});      // 2 input channels
    Tensor weights({2, 3, 3, 1});    // 2 groups, each has 1 kernel
    Tensor bias({2});
    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
        throw std::runtime_error("XNNPACK initialization failed");
    }
    pthreadpool_t pthreadpool_ = pthreadpool_create(0);
    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {3, 3},      // kernel_size
        {1, 1},
        {1, 1, 1, 1},
        {1, 1},
        2,  // groups = in_channels = out_channels (depthwise)
        pthreadpool_
    );

    EXPECT_EQ(output.shape(), std::vector<int>({1, 4, 4, 2}));
    ASSERT_EQ(output.shape()[3], 2);

    if (pthreadpool_) pthreadpool_destroy(pthreadpool_);
}
