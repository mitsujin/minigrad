#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

int main()
{
    xt::xarray<double> ar = xt::linspace<double>(0.0, 10.0, 12.0);
    ar.reshape({4,3});
    std::cout << ar << std::endl;
    
    std::cout << "Hello world\n";
}
