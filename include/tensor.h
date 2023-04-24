#include <concepts>
#include <span>
#include <memory>
#include <algorithm>

#include "tensor_shape_utils.h"

namespace MiniGrad
{
    template <typename T>
    concept Numeric = std::integral<T> || std::floating_point<T>;

        
    template <typename T> requires Numeric<T>
    class Tensor 
    {
    public:
        using DataPtr = std::shared_ptr<T[]>;

        // More resarch
        Tensor(const std::vector<size_t>& shape)
        : m_shapeHelper(shape)
        , m_data(new T[m_shapeHelper.size()], [](T* data) { delete[] data; })
        {
            std::fill_n(m_data.get(), m_shapeHelper.size(), T{});

            // TODO: Use a memory allocator here
        }

        Tensor(const std::vector<size_t>& shape, DataPtr data, int size)
        : m_shapeHelper(shape)
        , m_data(data)
        {
            if (m_shapeHelper.size() != size)
            {
                m_data = data;
            }
            else
            {
                throw std::runtime_error("Invalid shape");
            }
        }

        Tensor reshape(const std::vector<size_t>& shape)
        {
            return Tensor(shape, m_data, shape.size());
        }

        T& at(const std::vector<size_t>& indices)
        {
            size_t index = m_shapeHelper.calculateIndex(indices);
            return m_data.get()[index];
        }

        T at(const std::vector<size_t>& indices) const
        {
            size_t index = m_shapeHelper.calculateIndex(indices);
            return m_data.get()[index];
        }

        Tensor operator = (const Tensor& other) const
        {
            return Tensor(m_shapeHelper.shape(), m_data, m_shapeHelper.size());
        }

        std::span<T> data()
        {
            return std::span<T>(m_data.get(), m_shapeHelper.size());
        }

        size_t size() const
        {
            return m_shapeHelper.size();
        }

        template <typename U>
        friend std::ostream& operator << (std::ostream& oss, const Tensor<U>& t)
        {
            t.printTensor(oss);
            return oss;
        }

    private:
        void printTensor(std::ostream& oss, IndexArray indices = {}, size_t dim = 0) const
        {
            auto& shape = m_shapeHelper.shape();
            auto& strides = m_shapeHelper.strides();
            if (dim == shape.size() - 1) 
            {
                oss << "[";
                for (size_t i = 0u; i < shape[dim]; i++)
                {
                    indices.push_back(i);
                    oss << at(indices);
                    indices.pop_back();

                    if (i != shape[dim] - 1)
                    {
                        oss << ", ";
                    }
                }
                oss << "]";
            }
            else
            {
                oss << "[";
                for (size_t i = 0u; i < shape[dim]; i++)
                {
                    indices.push_back(i);
                    printTensor(oss, indices, dim + 1);
                    indices.pop_back();

                    if (i != shape[dim] - 1)
                    {
                        oss << ",\n";
                    }
                }
                oss << "]";
            }
        }

        TensorShapeHelper m_shapeHelper;
        DataPtr m_data = nullptr;
    };
}
