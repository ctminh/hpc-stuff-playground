cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(taskruntime-pred)

find_package(Torch REQUIRED)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(taskruntime-pred taskruntime-pred.cpp)
target_link_libraries(taskruntime-pred "${TORCH_LIBRARIES}")
set_property(TARGET taskruntime-pred PROPERTY CXX_STANDARD 14)
