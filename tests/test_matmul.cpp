#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(MatMulTest, BasicMultiplication) {
    Tensor a({2, 3});
    Tensor b({3, 2});

    // Fill tensors with known values
    std::fill(a.data().begin(), a.data().end(), 1.0f);
    std::fill(b.data().begin(), b.data().end(), 2.0f);

    MatMul matmul;
    Tensor result = matmul.compute(a, b);

    // Expect all elements to be 6 (1*2 + 1*2 + 1*2)
    for (float val : result.data()) {
        EXPECT_FLOAT_EQ(val, 6.0f);
    }
}
