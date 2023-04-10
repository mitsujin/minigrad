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

    private:
        IndexArray m_strides;
        IndexArray m_shape;
        size_t m_size;
    };
}

#endif /* TENSOR_SHAPE_UTILS_H */
