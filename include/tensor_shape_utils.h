#ifndef TENSOR_SHAPE_UTILS_H
#define TENSOR_SHAPE_UTILS_H

#include <vector>

namespace MiniGrad
{
    using IndexArray = std::vector<size_t>;

    class TensorShapeHelper
    {
    public:
        TensorShapeHelper(const IndexArray& shape);

        size_t calculateIndex(const IndexArray& indices) const;
        size_t size() const noexcept { return m_size; }
        const IndexArray& strides() const noexcept { return m_strides; }
        const IndexArray& shape() const noexcept { return m_shape; }

    private:
        IndexArray m_shape;
        IndexArray m_strides;
        size_t m_size;
    };
}

#endif /* TENSOR_SHAPE_UTILS_H */
