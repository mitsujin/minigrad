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

    private:
        T* m_data = nullptr;
        TensorShapeHelper m_shapeHelper;
    };
}
