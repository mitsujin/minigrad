cmake_minimum_required(VERSION 3.1)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(MiniGrad LANGUAGES CXX VERSION 0.0.1)

include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_path(XTENSOR_INCLUDE_DIR NAMES xtensor.hpp)
include_directories(${XTENSOR_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(lib/googletest)




