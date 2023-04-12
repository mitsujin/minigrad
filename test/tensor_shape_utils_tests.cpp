#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "tensor_shape_utils.h"

namespace MiniGrad
{
    TEST(TensorShapeHelper, TestCalculateIndex1D)
    {
        TensorShapeHelper ts({3});
        ASSERT_EQ(0, ts.calculateIndex({0}));
        ASSERT_EQ(1, ts.calculateIndex({1}));
        ASSERT_EQ(2, ts.calculateIndex({2}));

        EXPECT_THAT([&]() { ts.calculateIndex({3}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 1}); }, testing::Throws<std::out_of_range>());
    }

    TEST(TensorShapeHelper, TestCalculateIndex2D)
    {
        TensorShapeHelper ts({3, 2});
        ASSERT_EQ(0, ts.calculateIndex({0, 0}));
        ASSERT_EQ(1, ts.calculateIndex({0, 1}));
        ASSERT_EQ(2, ts.calculateIndex({1, 0}));
        ASSERT_EQ(3, ts.calculateIndex({1, 1}));
        ASSERT_EQ(4, ts.calculateIndex({2, 0}));
        ASSERT_EQ(5, ts.calculateIndex({2, 1}));

        EXPECT_THAT([&]() { ts.calculateIndex({0}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 1, 2}); }, testing::Throws<std::out_of_range>());
    }

    TEST(TensorShapeHelper, TestCalculateIndex3D)
    {
        TensorShapeHelper ts({3, 2, 4});
        ASSERT_EQ(0, ts.calculateIndex({0, 0, 0}));
        ASSERT_EQ(1, ts.calculateIndex({0, 0, 1}));
        ASSERT_EQ(2, ts.calculateIndex({0, 0, 2}));
        ASSERT_EQ(3, ts.calculateIndex({0, 0, 3}));

        ASSERT_EQ(4, ts.calculateIndex({0, 1, 0}));
        ASSERT_EQ(5, ts.calculateIndex({0, 1, 1}));
        ASSERT_EQ(6, ts.calculateIndex({0, 1, 2}));
        ASSERT_EQ(7, ts.calculateIndex({0, 1, 3}));

        ASSERT_EQ(8, ts.calculateIndex({1, 0, 0}));
        ASSERT_EQ(9, ts.calculateIndex({1, 0, 1}));
        ASSERT_EQ(10, ts.calculateIndex({1, 0, 2}));
        ASSERT_EQ(11, ts.calculateIndex({1, 0, 3}));

        ASSERT_EQ(12, ts.calculateIndex({1, 1, 0}));
        ASSERT_EQ(13, ts.calculateIndex({1, 1, 1}));
        ASSERT_EQ(14, ts.calculateIndex({1, 1, 2}));
        ASSERT_EQ(15, ts.calculateIndex({1, 1, 3}));

        ASSERT_EQ(16, ts.calculateIndex({2, 0, 0}));
        ASSERT_EQ(17, ts.calculateIndex({2, 0, 1}));
        ASSERT_EQ(18, ts.calculateIndex({2, 0, 2}));
        ASSERT_EQ(19, ts.calculateIndex({2, 0, 3}));

        ASSERT_EQ(20, ts.calculateIndex({2, 1, 0}));
        ASSERT_EQ(21, ts.calculateIndex({2, 1, 1}));
        ASSERT_EQ(22, ts.calculateIndex({2, 1, 2}));
        ASSERT_EQ(23, ts.calculateIndex({2, 1, 3}));

        EXPECT_THAT([&]() { ts.calculateIndex({0}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 1}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 1, 5}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 2, 5}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({3, 0, 0}); }, testing::Throws<std::out_of_range>());
        EXPECT_THAT([&]() { ts.calculateIndex({0, 0, 0, 0}); }, testing::Throws<std::out_of_range>());
    }
}
