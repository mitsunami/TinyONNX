#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(GlobalAveragePoolTest, BasicAveragePool) {
    Tensor input({1, 2, 2, 2});
    input.data() = {1, 2, 3, 4,  5, 6, 7, 8};

    Operators ops;
    Tensor output = ops.globalAveragePool(input);

    ASSERT_EQ(output.shape(), std::vector<int>({1, 2, 1, 1}));

    EXPECT_FLOAT_EQ(output.data()[0], (1+2+3+4)/4.0f); // channel 0
    EXPECT_FLOAT_EQ(output.data()[1], (5+6+7+8)/4.0f); // channel 1
}
