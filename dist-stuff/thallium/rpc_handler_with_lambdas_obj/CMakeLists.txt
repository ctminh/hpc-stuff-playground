# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_rpc_lambdas_obj)

# compile the code
add_executable(tl_rpc_lambd_server tl_rpc_lamb_obj_server.cpp)
target_link_libraries(tl_rpc_lambd_server -lthallium -lmargo -lmercury -labt)

add_executable(tl_rpc_lambd_client tl_rpc_lamb_obj_client.cpp)
target_link_libraries(tl_rpc_lambd_client -lthallium -lmargo -lmercury -labt)