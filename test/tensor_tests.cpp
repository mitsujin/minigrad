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
        t(0,1,0) = 2;
        //ASSERT_EQ(24u, t.size());
    }

    TEST(TensorTests, TestModify)
    {
        Tensor<int> t({3});
        t(1) = 3;
        t(2) = 4;

        ASSERT_EQ(3, t(1));
        ASSERT_EQ(4, t(2));
    }

    TEST(TensorTests, TestAssignment)
    {
        Tensor<int> t({4});
        t(0) = 4;
        t(2) = 2;

        Tensor<int> t2 = t;
        ASSERT_TRUE(t == t2);
    }

    TEST(TensorTests, TestReshape)
    {
        Tensor<int> t({1, 6});
        auto t2 = t.reshape({2, 3});
        t2(1, 2) = 5;
        ASSERT_EQ(t(5), t2(1,2));

        // points to same memory
        ASSERT_TRUE(t == t2);
    }
}
