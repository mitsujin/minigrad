#include "gtest/gtest.h"
#include "tensor.h"

namespace MiniGrad
{
    namespace 
    {
        template <typename T>
        void AssertCollection(std::span<T> expected, std::span<T> actual)
        {
            ASSERT_EQ(expected.size(), actual.size());
            int i = 0;
            for (auto& v : expected)
            {
                ASSERT_EQ(v, actual[i]);
                i++;
            }
        }

        template <typename T>
        void AssertCollection(std::vector<T> expected, std::span<T> actual)
        {
            std::span<T> expectedS(expected.data(), expected.size());
            AssertCollection(expectedS, actual);
        }

    }
    TEST(TensorTests, TestCreation)
    {
        Tensor<int> t({2, 3, 4});
        t.at({0,1,0}) = 2;
        std::cout << t;
        ASSERT_EQ(24u, t.size());
    }

    TEST(TensorTests, TestModify)
    {
        Tensor<int> t({3});
        t.at({1}) = 3;
        t.at({2}) = 4;

        std::vector<int>expected({0, 3, 4});
        AssertCollection(expected, t.data());
    }

    TEST(TensorTests, TestAssignment)
    {
        Tensor<int> t({4});
        t.at({0}) = 4;
        t.at({2}) = 2;

        Tensor<int> t2 = t;

        AssertCollection(t.data(), t2.data());
    }

    TEST(TensorTests, TestReshape)
    {
        Tensor<int> t({6});
        auto t2 = t.reshape({2, 3});
        t2.at({1, 2}) = 5;

        std::vector<int>expected({0, 0, 0, 0, 0, 5});
        AssertCollection(expected, t.data());
        AssertCollection(expected, t2.data());
    }
}
