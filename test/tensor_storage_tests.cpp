#include "gtest/gtest.h"

#include "tensor_storage.h"

namespace MiniGrad
{
    TEST(TensorStorageTests, TestConstructor)
    {
        TensorStorage<int> s(10);
        ASSERT_TRUE(true);
    }
