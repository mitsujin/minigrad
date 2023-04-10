#include "gtest/gtest.h"
#include "tensor.h"

namespace MiniGrad
{
    TEST(TensorTests, TestCreation)
    {
        Tensor<int> t({2, 3, 4});
        std::cout << t;
        ASSERT_EQ(24u, t.size());
    }
}
