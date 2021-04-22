cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS -std=c++14)
project(singleAlignCorre)

find_package(Ceres REQUIRED)

# singleAlignCorre
add_executable(singleAlignCorre singleAlignCorre.cc)
target_link_libraries(singleAlignCorre Ceres::ceres)