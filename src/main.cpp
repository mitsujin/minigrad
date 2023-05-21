#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xstrided_view.hpp>

#include "tensor.h"

using namespace MiniGrad;

int main()
{
    xt::xarray<double> ar = xt::linspace<double>(0.0, 10.0, 12.0);

    ar.reshape({4,3});
    std::cout << ar << std::endl;
    std::cout << xt::adapt(ar.shape()) << std::endl;

    Tensor<double> tensor = Tensor<double>::fromShape({4,4});
    std::cout << tensor << std::endl;

    /*
    auto shape = {1,16};
    xt::xview<int> t2 = xt::reshape_view(test, {1,16});
    t2(0) = 2;
    std::cout << t2 << std::endl;
    std::cout << test << std::endl;
    */
    xt::xarray<int> test = xt::xarray<int>::from_shape({4,4});
    using shape_type = std::vector<size_t>;
    shape_type shape = {1, 16};
    using view_type = decltype(xt::reshape_view(std::declval<xt::xarray<int>&>(), std::declval<shape_type&>()));
    view_type v = xt::reshape_view(test, shape);
    v(0) = 2;
    std::cout << test << std::endl;
    std::cout << v << std::endl;
    const auto& v2 = test;
    std::cout << v2;


    /*
    using view_type = xt::xstrided_view<
        xt::xclosure_t<xt::xarray<int>>,
        std::vector<size_t>,
        xt::layout_type::dynamic,
        xt::detail::flat_adaptor_getter<xt::xclosure_t<xt::xarray<int>>>;
        */
    
        
    std::cout << "Hello world\n";
}
