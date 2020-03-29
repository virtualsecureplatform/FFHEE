cmake_minimum_required(VERSION 3.10)

#project(cuexternalproduct LANGUAGES CUDA CXX)
project(cuexternalproducttest LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-std=c++14 -arch=sm_70 --ptxas-options=-v -ccbin clang++-8")

add_executable(cuexternalproducttest main.cu)
target_link_libraries(cuexternalproducttest PRIVATE tfhe++)
target_include_directories(cuexternalproducttest
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/mulfft
    ${CMAKE_CURRENT_LIST_DIR}/../../TFHEpp/include
    ${CMAKE_CURRENT_LIST_DIR}/../../TFHEpp/thirdparties/randen
    ${CMAKE_CURRENT_LIST_DIR}/../../TFHEpp/thirdparties/spqlios
    )

#add_executable(convolutioncpu convolutioncpu.cpp)
#target_include_directories(convolutioncpu
#    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
#    )
#target_compile_options(convolutioncpu
#    PUBLIC -g
#    )