#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(Conv2DTest, BasicConvolution) {
    Tensor input({1, 3, 224, 224});
    Tensor weights({16, 3, 3, 3});
    Tensor bias({16});

    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;
    Tensor output = ops.conv2d(input, weights, bias, 1, 1);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 224);
    EXPECT_EQ(output.shape()[3], 224);
}
