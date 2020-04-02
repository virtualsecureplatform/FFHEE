cmake_minimum_required(VERSION 3.10)

#project(cuexternalproduct LANGUAGES CUDA CXX)
project(cugatebootstrapping LANGUAGES CUDA CXX)

find_package(OpenMP)

set(CMAKE_CUDA_FLAGS "-std=c++14 -arch=sm_70 --ptxas-options=-v -ccbin clang++-8")

add_executable(cugatebootstrapping main.cu)
target_link_libraries(cugatebootstrapping PRIVATE tfhe++ OpenMP::OpenMP_CXX)
target_include_directories(cugatebootstrapping
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/spculios
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