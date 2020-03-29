cmake_minimum_required(VERSION 3.10)
project(pagelock LANGUAGES CUDA CXX)

add_executable(pagelock pagelock.cu)
target_include_directories(pagelock
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )
