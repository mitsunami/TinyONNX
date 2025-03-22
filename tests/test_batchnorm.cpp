#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(BatchNormTest, BasicBatchNorm) {
    Tensor input({1, 3, 4, 4});
    Tensor scale({3});
    Tensor bias({3});
    Tensor mean({3});
    Tensor var({3});

    input.fillRandom();
    scale.fillRandom();
    bias.fillRandom();
    mean.fillRandom();
    var.fillRandom();

    Operators ops;
    float epsilon = 1e-5f;
    Tensor output = ops.batchNorm(input, scale, bias, mean, var, epsilon);

    EXPECT_EQ(output.shape(), input.shape());
}
