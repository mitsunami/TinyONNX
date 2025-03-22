#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(ReLUTest, BasicReLU) {
    Tensor input({3});
    input.data() = {-1.0f, 0.0f, 1.0f};

    Operators ops;
    Tensor output = ops.relu(input);

    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 1.0f);
}
