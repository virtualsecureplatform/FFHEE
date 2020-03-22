cmake_minimum_required(VERSION 3.10)

project(ifft LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-std=c++14 --expt-relaxed-constexpr --ptxas-options=-v")

add_executable(ifft ifft.cu)
target_include_directories(ifft
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )

add_executable(ifftcpu ifftcpu.cpp)
target_include_directories(ifftcpu
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )
target_compile_options(ifftcpu
    PUBLIC -g
    )