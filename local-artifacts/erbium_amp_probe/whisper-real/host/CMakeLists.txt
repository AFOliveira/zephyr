cmake_minimum_required(VERSION 3.21)

project(whisper_tail_host_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/lib/cmake")

find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)

add_executable(whisper_tail_parallel_argbuf whisper_tail_parallel_argbuf.cpp)

target_link_libraries(whisper_tail_parallel_argbuf
    PRIVATE
        deviceLayer::deviceLayer
        runtime::etrt
)
