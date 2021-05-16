# require a minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# declare the project name
project(simple_sum)

# create a share library
add_library(sum_func SHARED sum_function.cpp)
target_include_directories(sum_func PUBLIC ./)

# create an executable which needs to call the shared lib
add_executable(main ./main.cpp)

# link the executable with the shared lib
target_link_libraries(main PRIVATE sum_func)