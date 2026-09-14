cmake_minimum_required(VERSION 4.0)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME eve
    URL https://github.com/jfalcou/eve/archive/refs/tags/v2022.09.1.zip
    VERSION 2022.09.1
    OPTIONS "EVE_BUILD_TEST OFF"
    "EVE_BUILD_BENCHMARKS OFF"
    "EVE_BUILD_DOCUMENTATION OFF"
)

find_package(OpenCV 4.2 REQUIRED)

add_executable(lab2 main.cpp
        blend.cpp
        blend.h
        hist.cpp
        hist.h
        benchmark.cpp
        benchmark.h)

target_compile_definitions(lab2 PRIVATE EVE_USE_NEON=1)

target_link_libraries(lab2 PRIVATE ${OpenCV_LIBS})