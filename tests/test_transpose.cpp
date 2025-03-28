// test_transpose.cpp
#include <gtest/gtest.h>
#include "tensor.h"
#include "operators.h"

TEST(TransposeTest, BasicPermuteNHWCtoNCHW) {
    // Input NHWC tensor (shape [1, 2, 2, 3])
    Tensor input({1, 2, 2, 3}, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,

        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    });

    // Permutation NHWC -> NCHW
    std::vector<int> perm = {0, 3, 1, 2};

    Operators ops;
    Tensor output = ops.transpose(input, perm);

    // Expected NCHW shape: [1, 3, 2, 2]
    std::vector<int> expected_shape = {1, 3, 2, 2};
    ASSERT_EQ(output.shape(), expected_shape);

    std::vector<float> expected_data = {
        1.0, 4.0,
        7.0, 10.0,

        2.0, 5.0,
        8.0, 11.0,

        3.0, 6.0,
        9.0, 12.0
    };

    EXPECT_EQ(output.data(), expected_data);
}

TEST(TransposeTest, BasicPermuteNCHWtoNHWC) {
    // Input NCHW tensor (shape [1, 2, 2, 3])
    Tensor input({1, 3, 2, 2}, {
        1.0, 4.0,
        7.0, 10.0,

        2.0, 5.0,
        8.0, 11.0,

        3.0, 6.0,
        9.0, 12.0
    });

    // Permutation NCHW -> NHWC
    std::vector<int> perm = {0, 2, 3, 1};

    Operators ops;
    Tensor output = ops.transpose(input, perm);

    // Expected NHWC shape: [1, 2, 2, 3]
    std::vector<int> expected_shape = {1, 2, 2, 3};
    ASSERT_EQ(output.shape(), expected_shape);

    std::vector<float> expected_data = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,

        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    EXPECT_EQ(output.data(), expected_data);
}