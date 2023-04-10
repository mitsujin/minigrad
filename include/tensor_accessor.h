#ifndef TENSOR_ACCESSOR_H
#define TENSOR_ACCESSOR_H

namespace MiniGrad
{
    <template typename T, N>
    class TensorAccessorBase
    {
    public:
        TensorAccessor(T* dataPtr, int* strides, int* sizes)
            : dataPtr(dataPtr)
            , strides(strides)
            , sizes(sizes) {

        }

    protected:
        int* strides;
        int* sizes;
        T* dataPtr;
    };

    <template typename T, N>
    class TensorAccessor : public TensorAccessorBase<T, N>
    {
    public:
        using TensorAccessorBase;

        const TensorAccessor<T, N-1> operator[](size_t index) const {
            return TensorAccessor<T, N-1>(dataPtr + strides[0] * index, strides+1, sizes+1);
        }
        TensorAccessor<T, N-1> operator[](size_t index) {
            return TensorAccessor<T, N-1>(dataPtr + strides[0] * index, strides+1, sizes+1);
        }
    };

    <template typename T, N>
    class TensorAccessor<T, 1> : public TensorAccessorBase<T, N>
    {
        TensorAccessor(T* dataPtr, int* strides, int* sizes)
            : dataPtr(dataPtr)
            , strides(strides)
            , sizes(sizes) {

        }

        const T& operator[](size_t index) const {
            return dataPtr[strides[0] * index];
        }

        T& operator[](size_t index) {
            return dataPtr[strides[0] * index];
        }
    };
}

#endif /* TENSOR_ACCESSOR_H */
