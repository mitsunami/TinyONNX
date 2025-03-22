#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(Conv2DTest, BasicNoPaddingStride1) {
    Tensor input({1, 1, 3, 3});  // Batch=1, Channels=1, 3x3
    Tensor weights({1, 1, 2, 2}); // Out_channels=1, In_channels=1, 2x2
    Tensor bias({1});

    input.data() = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    weights.data() = {
        1, 0,
        0, -1
    };
    bias.data()[0] = 0.0f;

    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {1, 1},      // stride
        {0, 0, 0, 0},// padding (top, left, bottom, right)
        {1, 1},      // dilation
        1            // groups
    );

    ASSERT_EQ(output.shape(), std::vector<int>({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.data()[0], -4);  // top-left
    EXPECT_FLOAT_EQ(output.data()[1], -4);  // top-right
    EXPECT_FLOAT_EQ(output.data()[2], -4);  // bottom-left
    EXPECT_FLOAT_EQ(output.data()[3], -4);  // bottom-right
}

TEST(Conv2DTest, Stride2Padding1) {
    Tensor input({1, 1, 4, 4});
    Tensor weights({1, 1, 3, 3});
    Tensor bias({1});

    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {2, 2},
        {1, 1, 1, 1},
        {1, 1},
        1
    );

    // Output size formula: (in + pad*2 - kernel) / stride + 1
    // → (4 + 2 - 3)/2 + 1 = 2
    EXPECT_EQ(output.shape(), std::vector<int>({1, 1, 2, 2}));
}

TEST(Conv2DTest, Dilation2) {
    Tensor input({1, 1, 7, 7});
    Tensor weights({1, 1, 3, 3});
    Tensor bias({1});
    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {1, 1},
        {0, 0, 0, 0},
        {2, 2}, // dilated kernel size becomes 5x5
        1
    );

    EXPECT_EQ(output.shape(), std::vector<int>({1, 1, 3, 3}));
}

TEST(Conv2DTest, DepthwiseGroupConv) {
    Tensor input({1, 2, 4, 4});      // 2 input channels
    Tensor weights({2, 1, 3, 3});    // 2 groups, each has 1 kernel
    Tensor bias({2});
    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;
    Tensor output = ops.conv2d(
        input, weights, bias,
        {1, 1},
        {1, 1, 1, 1},
        {1, 1},
        2  // groups = in_channels = out_channels (depthwise)
    );

    EXPECT_EQ(output.shape(), std::vector<int>({1, 2, 4, 4}));
}
