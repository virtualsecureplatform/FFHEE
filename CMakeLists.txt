cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CXX_FLAGS "-march=native -O3 -std=c++17 -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O3 -arch=sm_70")
set(CMAKE_CXX_STANDARD 14)

project(FFHEE LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

#add_subdirectory(TFHEpp)
#target_link_libraries(FFHEE INTERFACE tfhe++)

#add_subdirectory(test)
add_subdirectory(playground)