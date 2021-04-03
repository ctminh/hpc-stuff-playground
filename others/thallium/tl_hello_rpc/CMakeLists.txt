# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_rpc_hello)

# compile the code
add_executable(tl_rpc_server tl_rpc_server.cpp)
target_link_libraries(tl_rpc_server -lthallium -lmargo -lmercury -labt)

add_executable(tl_rpc_client tl_rpc_client.cpp)
target_link_libraries(tl_rpc_client -lthallium -lmargo -lmercury -labt)