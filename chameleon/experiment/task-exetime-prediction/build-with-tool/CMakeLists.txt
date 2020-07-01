cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cham_tool)

find_package(Torch REQUIRED)
add_executable(cham_tool tool.cpp)
target_link_libraries(cham_tool "${TORCH_LIBRARIES}")
set_property(TARGET cham_tool PROPERTY CXX_STANDARD 11)