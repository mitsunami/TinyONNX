#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(SoftmaxTest, BasicSoftmax) {
    Tensor input({4});
    input.data() = {1.0f, 2.0f, 3.0f, 4.0f};

    Operators ops;
    Tensor output = ops.softmax(input);

    float sum = 0.0f;
    for (float val : output.data())
        sum += val;

    // Verify sum is close to 1
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
