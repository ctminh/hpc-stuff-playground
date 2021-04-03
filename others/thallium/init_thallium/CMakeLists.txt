# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_init)

# compile the code
add_executable(tl_server server.cpp)
target_link_libraries(tl_server thallium -lmargo -lmercury -labt)

add_executable(tl_client client.cpp)
target_link_libraries(tl_client thallium -lmargo -lmercury -labt)