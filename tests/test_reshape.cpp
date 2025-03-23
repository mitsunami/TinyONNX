#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(ReshapeTest, BasicReshape) {
    Tensor input({1, 2, 2, 2});
    input.data() = {1, 2, 3, 4, 5, 6, 7, 8};

    Operators ops;

    Tensor output = ops.reshape(input, {1, -1}); // Reshape to [1, 8]

    EXPECT_EQ(output.shape(), std::vector<int>({1, 8}));
    EXPECT_EQ(output.data(), input.data());
}
