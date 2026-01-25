cmake_minimum_required(VERSION 4.0)
project(lab3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

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

add_executable(lab3 main.cpp
        src/blend.cpp
        src/blend.h
        src/hist.cpp
        src/hist.h
        src/benchmark.cpp
        src/benchmark.h
        src/ThreadPool.cpp
        src/ThreadPool.h
)

target_compile_definitions(lab3 PRIVATE EVE_USE_NEON=1)

target_link_libraries(lab3 PRIVATE ${OpenCV_LIBS})