cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cxx_load_pytorch)

find_package(Torch REQUIRED)

add_executable(cxx_load_pytorch cxx_load_pytorch.cpp)
target_link_libraries(cxx_load_pytorch "${TORCH_LIBRARIES}")
set_property(TARGET cxx_load_pytorch PROPERTY CXX_STANDARD 14)