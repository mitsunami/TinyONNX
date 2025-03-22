#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(AddTest, ElementwiseAddition) {
    Tensor a({2, 2});
    Tensor b({2, 2});

    a.data() = {1.0f, 2.0f, 3.0f, 4.0f};
    b.data() = {4.0f, 3.0f, 2.0f, 1.0f};

    Operators ops;
    Tensor output = ops.add(a, b);

    EXPECT_EQ(output.data(), std::vector<float>({5.0f, 5.0f, 5.0f, 5.0f}));
}
