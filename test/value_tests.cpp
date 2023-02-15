#include "gtest/gtest.h"
#include "value.h"
#include <fstream>
#include <iostream>

namespace SimpleFlow
{
    TEST(ValueTests, TestAddition)
    {
        auto a = Value(2);
        auto b = Value(3);
        auto c = a + b;

        c.printGraph(std::cout);

        ASSERT_EQ(c.get(), 5);
    }

    TEST(ValueTests, TestAddition2)
    {
        auto a = Value(2);
        auto b = Value(3);
        auto c = b + a;

        ASSERT_EQ(c.get(), 5);
    }

    TEST(ValueTests, TestSub)
    {
        auto a = Value(2);
        auto b = Value(3);
        auto c = a - b;

        ASSERT_EQ(c.get(), -1);
    }

    TEST(ValueTests, TestSub2)
    {
        auto a = Value(2);
        auto b = Value(3);
        auto c = b - a;

        ASSERT_EQ(c.get(), 1);
    }

    TEST(ValueTests, TestSub3)
    {
        auto a = Value(2);
        auto b = 3.0;
        auto c = a - b;

        ASSERT_EQ(c.get(), -1);
    }

    TEST(ValueTests, TestMul)
    {
        auto a = Value(3);
        auto b = Value(10);
        auto c = a * b;

        ASSERT_EQ(c.get(), 30);
    }

    TEST(ValueTests, TestAddGrad)
    {
        auto a = Value(2.0);
        auto b = Value(5.0);
        auto c = a + b;

        c.backward();
        auto f = [](double x1, double x2)
        {
            return x1 + x2;
        };
        const float d = 0.0001;
        auto x1 = a.get();
        auto x2 = b.get();

        auto gradA = (f(x1 + d, x2) - f(x1, x2)) / d;
        auto gradB = (f(x1, x2 + d) - f(x1, x2)) / d;

        EXPECT_FLOAT_EQ(gradA, a.grad());
        EXPECT_FLOAT_EQ(gradB, b.grad());
    }

    TEST(ValueTests, TestAddWithLiteral)
    {
        auto a = Value(2.0);
        auto c = a + 5.0;
        c.backward();
        auto f = [](double x1, double x2)
        {
            return x1 + x2;
        };
        const float d = 0.0001;
        auto x1 = a.get();

        auto gradA = (f(x1 + d, 5.0) - f(x1, 5.0)) / d;

        EXPECT_FLOAT_EQ(gradA, a.grad());
    }

    TEST(ValueTests, TestMulGrad)
    {
        auto a = Value(2.0);
        auto b = Value(5.0);
        auto c = a * b;

        c.backward();
        auto f = [](double x1, double x2)
        {
            return x1 * x2;
        };
        const float d = 0.0001;
        auto x1 = a.get();
        auto x2 = b.get();

        auto gradA = (f(x1 + d, x2) - f(x1, x2)) / d;
        auto gradB = (f(x1, x2 + d) - f(x1, x2)) / d;

        EXPECT_FLOAT_EQ(gradA, a.grad());
        EXPECT_FLOAT_EQ(gradB, b.grad());
    }

    TEST(ValueTests, TestMulWithLiteral)
    {
        auto a = Value(2.0);
        auto c = a * 5.0;
        c.backward();
        auto f = [](double x1, double x2)
        {
            return x1 * x2;
        };
        const float d = 0.0001;
        auto x1 = a.get();

        auto gradA = (f(x1 + d, 5.0) - f(x1, 5.0)) / d;

        EXPECT_FLOAT_EQ(gradA, a.grad());
    }

    TEST(ValueTest, TestMultiExpression)
    {
        auto a = Value(2.0, "a");
        auto b = Value(-3.0, "b");
        auto c = Value(10.0, "c");
        auto e = a*b; e.setLabel("e");
        auto d = e + c; d.setLabel("d");
        auto f = Value(-2.0, "f");
        auto L = d * f; L.setLabel("L");

        L.backward();
        std::ofstream out("graph.txt");
        L.printGraph(out);
        out.close();

        auto F = [](double a, double b, double c) {
            double e = a * b;
            double d = e + c;
            double f = -2.0;
            double L = d * f;
            return L;
        };

        auto x1 = a.get();
        auto x2 = b.get();
        auto x3 = c.get();

        const double ep = 0.0001;
        auto gradA = (F(x1 + ep, x2, x3) - F(x1, x2, x3)) / ep;
        auto gradB = (F(x1, x2 + ep, x3) - F(x1, x2, x3)) / ep;
        auto gradC = (F(x1, x2, x3 + ep) - F(x1, x2, x3)) / ep;

        EXPECT_FLOAT_EQ(F(x1,x2,x3), L.get());

        EXPECT_FLOAT_EQ(gradA, a.grad());
        EXPECT_FLOAT_EQ(gradB, b.grad());
        EXPECT_FLOAT_EQ(gradC, c.grad());
    }

    TEST(ValueTest, TestTanh)
    {
        auto a = Value(0.8814, "a");
        auto b = a.tanh();
        b.backward();

        EXPECT_NEAR(0.5, a.grad(), 1e-3);
    }

    TEST(ValueTest, TestPow)
    {
        auto x = Value(2.0, "x");
        auto y = x.pow(3.0);
        y.backward();

        // x**3 = 3 * x**2
        EXPECT_DOUBLE_EQ(12.0, x.grad());
    }

    TEST(ValueTest, TestDiv)
    {
        auto x = Value(100.0, "x");
        auto y = x / Value(5.0);

        y.backward();

        EXPECT_DOUBLE_EQ(0.2, x.grad());
    }

    TEST(ValueTest, TestDivL)
    {
        auto x = Value(300.0, "x");
        auto y = x / 6.0;

        y.backward();

        EXPECT_DOUBLE_EQ(1/6.0, x.grad());
    }

    TEST(ValueTest, TestMLP)
    {
        auto x1 = Value(2.0, "x1"); 
        auto x2 = Value(0.0, "x2"); 

        // weights
        auto w1 = Value(-3.0, "w1");
        auto w2 = Value(1.0, "w2");

        // bias
        auto b = Value(6.8813735870195432, "b");

        auto n = (x1 * w1 + x2 * w2) + b;
        n.setLabel("n");

        auto o = n.tanh(); o.setLabel("o");
        o.backward();

        std::ofstream out("graph_mlp.txt");
        o.printGraph(out);
        out.close();

        EXPECT_DOUBLE_EQ(-1.5, x1.grad());
        EXPECT_DOUBLE_EQ(1.0, w1.grad());
        EXPECT_DOUBLE_EQ(0.5, x2.grad());
        EXPECT_DOUBLE_EQ(0.0, w2.grad());
        EXPECT_DOUBLE_EQ(0.5, b.grad());
    }
}
