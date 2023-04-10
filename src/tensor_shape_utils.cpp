#include "tensor_shape_utils.h"
#include <stdexcept>

namespace MiniGrad
{
    namespace
    {
        size_t calculateSize(const IndexArray& shape) 
        {
            size_t size = shape.empty() ? 0u : 1u;
            for (auto dim : shape)
            {
                size *= dim;
            }
            return size;
        }
    }

    TensorShapeHelper::TensorShapeHelper(const IndexArray& shape)
    : m_strides(shape.size(), 1)
    , m_size(calculateSize(shape))
    {
        if (m_strides.size() > 1)
        {
            for (int i = m_strides.size() - 2; i >= 0; i--)
            {
                m_strides[i] = m_strides[i+1] * shape[i+1];
            }
        }
    }

    size_t TensorShapeHelper::calculateIndex(const IndexArray& indices) const 
    {
        if (indices.size() != m_shape.size())
        {
            throw std::out_of_range("Indices are out of range! Does not match the Tensor shape");
        }

        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++)
        {
            if (index >= m_shape[i])
            {
                throw std::out_of_range("Index is out of range of the shape");
            }
            index += m_strides[i] * indices[i];
        }

        return index;
    }
}
