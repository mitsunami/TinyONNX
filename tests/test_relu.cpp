#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(ReLUTest, BasicReLU) {
    Tensor t({3});
    t.data() = {-1.0f, 0.0f, 1.0f};

    ReLU relu;
    relu.compute(t);

    EXPECT_FLOAT_EQ(t.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 1.0f);
}
