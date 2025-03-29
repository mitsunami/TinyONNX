#include <gtest/gtest.h>
#include "operators.h"
#include "tensor.h"

TEST(GlobalAveragePoolTest, BasicAveragePool) {
    Tensor input({1, 2, 2, 3}, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    });

    Operators ops;
    Tensor output = ops.globalAveragePool(input);

    Tensor expected_output({1, 1, 1, 3}, {
        (1.0 + 4.0 + 7.0 + 10.0) / 4,
        (2.0 + 5.0 + 8.0 + 11.0) / 4,
        (3.0 + 6.0 + 9.0 + 12.0) / 4
    });

    EXPECT_EQ(output.shape(), expected_output.shape());

    for (size_t i = 0; i < output.data().size(); ++i) {
        EXPECT_NEAR(output.data()[i], expected_output.data()[i], 1e-5);
    }
}
