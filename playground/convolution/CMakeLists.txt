cmake_minimum_required(VERSION 3.10)

project(convolution LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-std=c++14 -arch=sm_70 --ptxas-options=-v")

add_executable(convolution main.cu)
target_include_directories(convolution
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/mulfft
    )

add_executable(convolutioncpu convolutioncpu.cpp)
target_include_directories(convolutioncpu
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )
target_compile_options(convolutioncpu
    PUBLIC -g
    )