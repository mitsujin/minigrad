#include <concepts>
#include <span>
#include "tensor_shape_utils.h"

namespace MiniGrad
{
    template <typename T>
    concept Numeric = std::integral<T> || std::floating_point<T>;
        
    template <typename T> requires Numeric<T>
    class Tensor 
    {
    public:
        // More resarch
        Tensor(const std::vector<size_t>& shape)
        : m_shapeHelper(shape)
        {
            // TODO: Use a memory allocator here
            m_data = new T[m_shapeHelper.size()];
        }

        ~Tensor()
        {
            delete[] m_data;
        }

        T& at(const std::vector<size_t>& indices)
        {
            size_t index = m_shapeHelper.calculateIndex(indices);
            return m_data[index];
        }

        T at(const std::vector<size_t>& indices) const
        {
            size_t index = m_shapeHelper.calculateIndex(indices);
            return m_data[index];
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

        T* m_data = nullptr;
        TensorShapeHelper m_shapeHelper;
    };
}
