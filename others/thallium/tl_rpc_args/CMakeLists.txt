# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_rpc_args_transf)

# compile the code
add_executable(tl_args_server tl_args_server.cpp)
target_link_libraries(tl_args_server -lthallium -lmargo -lmercury -labt)

add_executable(tl_args_client tl_args_client.cpp)
target_link_libraries(tl_args_client -lthallium -lmargo -lmercury -labt)