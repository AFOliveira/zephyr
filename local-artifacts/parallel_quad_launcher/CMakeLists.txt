cmake_minimum_required(VERSION 3.16)
project(parallel_quad_launcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "/home/afonso/et/lib/cmake")

find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(esperantoTrace REQUIRED)

set(ET_PLATFORM_ROOT "/home/afonso/et-platform-vidas" CACHE PATH "et-platform source root")

add_library(etsoc_gpsdk STATIC
  "${ET_PLATFORM_ROOT}/gp-sdk/host/sdk/src/lib/GenericLauncher.cpp"
)
target_include_directories(etsoc_gpsdk PUBLIC
  "${ET_PLATFORM_ROOT}/gp-sdk/host/sdk/include"
)
target_link_libraries(etsoc_gpsdk
  PUBLIC
    deviceLayer::deviceLayer
    runtime::etrt
  PRIVATE
    esperantoTrace::et_trace
)

add_executable(parallel_quad_shire_launcher
  ParallelQuadShire.cpp
)
target_link_libraries(parallel_quad_shire_launcher PRIVATE etsoc_gpsdk pthread)
