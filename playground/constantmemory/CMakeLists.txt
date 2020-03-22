cmake_minimum_required(VERSION 3.10)
project(constantmemory LANGUAGES CUDA CXX)

add_executable(constantmemory constantmemory.cu)
target_include_directories(constantmemory
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )
