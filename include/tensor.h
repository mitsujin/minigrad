#include <concepts>
#include <span>
#include <memory>
#include <algorithm>
#include <sstream>

#include <xtensor/xarray.hpp>
#include "tensor_shape_utils.h"

namespace MiniGrad
{
    template <typename T>
    concept Numeric = std::integral<T> || std::floating_point<T>;

    using shape_type = std::vector<size_t>;

    template <typename T> requires Numeric<T>
    class Tensor 
    {
    public:
        using storage_type = xt::xarray<T, xt::layout_type::dynamic>;
        using data_ptr_type = std::shared_ptr<storage_type>;
        using shape_type = storage_type::shape_type;

        //template <typename U> requires Numeric<U>
        static Tensor<T> fromShape(shape_type shape)
        {
            return Tensor<T>(storage_type::from_shape(shape));
        }

        Tensor(storage_type&& data) 
        : m_data(new storage_type(std::move(data)))
        , m_view(xt::reshape_view(*m_data, m_data->shape()))
        {
        }

        Tensor(const Tensor& o)
        : m_data(o.m_data)
        , m_view(xt::reshape_view(*o.m_data, o.m_data->shape()))
        {
            std::cout << "Copy constructor";
        }

        Tensor(data_ptr_type data, const shape_type& shape)
        : m_data(data)
        , m_view(xt::reshape_view(*m_data, shape))
        {
        }

        Tensor reshape(const shape_type& shape)
        {
            return Tensor(m_data, shape);
        }

        shape_type shape() const
        {
            return m_view.shape();
        }

        template <typename... Args>
        inline T operator()(Args... args) const 
        {
            return (m_view)(args...);
        }

        template <typename... Args>
        inline T& operator()(Args... args) 
        {
            return (m_view)(args...);
        }

        Tensor operator = (const Tensor& other) const
        {
            std::cout << "Operatror = ";
            return Tensor(m_data, m_data.shape());
        }

        bool operator == (const Tensor& other) const
        {
            return *m_data == *other.m_data;
        }

        template <typename U>
        friend std::ostream& operator << (std::ostream& oss, const Tensor<U>& t)
        {
            //oss << t.m_view;
            return oss;
        }

        // Gradient helpers
        bool requiresGrad() const 
        {
            return m_requiresGrad;
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << m_view;

            return oss.str();
        }

    private:
        // Underlying data
        std::shared_ptr<storage_type> m_data;

        using view_type = decltype(xt::reshape_view(std::declval<storage_type&>(), std::declval<shape_type&>()));
        view_type m_view;

        bool m_isView = false;
        bool m_requiresGrad = false;
    };

}
